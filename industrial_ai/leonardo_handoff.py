from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .hashing import file_sha256
from .leonardo_bundle import DEFAULT_BUNDLE_MANIFEST_PATH, DEFAULT_BUNDLE_PATH, verify_bundle
from .paths import PROJECT_ROOT


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_sidecar_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    return text.split()[0] if text else ""


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _handoff_commands(readiness: dict[str, object]) -> list[str]:
    commands: list[str] = []
    for key in ("full_pipeline", "split_jobs_with_dependencies"):
        rows = readiness.get("commands", {})
        if isinstance(rows, dict):
            commands.extend(str(command) for command in _as_list(rows.get(key)))
    commands.extend(str(command) for command in _as_list(readiness.get("verification_commands")))
    return commands


def _effective_source_bundle_paths(bundle_path: Path, manifest_path: Path) -> tuple[Path, Path]:
    effective_bundle = bundle_path
    effective_manifest = manifest_path
    root_bundle = PROJECT_ROOT / DEFAULT_BUNDLE_PATH.name
    root_manifest = PROJECT_ROOT / DEFAULT_BUNDLE_MANIFEST_PATH.name
    if bundle_path == DEFAULT_BUNDLE_PATH and not bundle_path.exists() and root_bundle.exists():
        effective_bundle = root_bundle
    if manifest_path == DEFAULT_BUNDLE_MANIFEST_PATH and not manifest_path.exists() and root_manifest.exists():
        effective_manifest = root_manifest
    return effective_bundle, effective_manifest


def _write_checklist(path: Path, payload: dict[str, object]) -> None:
    bundle_sha = str(payload.get("bundle_sha256", "") or "")
    upload_files = [Path(str(item)).name for item in _as_list(payload.get("upload_files"))]
    handoff_commands = [str(item) for item in _as_list(payload.get("handoff_commands"))]
    launch_commands = [
        command
        for command in handoff_commands
        if command.startswith("sbatch ") or command.startswith("PROBE_JOB=")
        or command.startswith("GEN_JOB=") or command.startswith("TRAIN_JOB=")
        or command.startswith("FINAL_JOB=") or command.startswith("echo ")
    ]
    verification_commands = [
        command
        for command in handoff_commands
        if command.startswith("python -m ")
    ]
    lines = [
        "# Leonardo Handoff Checklist",
        "",
        f"- Bundle SHA256: `{bundle_sha}`",
        f"- Handoff audit passed: `{str(payload.get('passed')).lower()}`",
        "",
        "## Upload Files",
        "",
    ]
    lines.extend(f"- `{item}`" for item in upload_files)
    lines.extend([
        "",
        "## Verify Uploaded Bundle On Leonardo",
        "",
        "```bash",
        "sha256sum -c leonardo_transfer_packet.zip.sha256  # if using the transfer packet",
        "unzip -o leonardo_transfer_packet.zip              # if using the transfer packet",
        "sha256sum -c leonardo_source_bundle.zip.sha256",
        "unzip -o leonardo_source_bundle.zip",
        "python -m industrial_ai.leonardo_handoff --verify-transfer-packet leonardo_transfer_packet.zip --verify-fresh-unpack",
        "python -m industrial_ai.leonardo_bundle --verify-bundle leonardo_source_bundle.zip --sidecar leonardo_source_bundle.zip.sha256",
        "python -m industrial_ai.leonardo_bundle --verify-root . --manifest leonardo_source_bundle_manifest.json",
        "```",
        "",
        "## Prepare Python Environment",
        "",
        "Use the event-provided module stack or a virtual environment that provides CUDA-enabled PyTorch.",
        "",
        "```bash",
        "python --version",
        "python -m pip install -r requirements.txt",
        "python -m industrial_ai.preflight --require-torch --require-cuda --out artifacts/preflight_leonardo_environment.json",
        "```",
        "",
        "## Stage Official Eval Files",
        "",
        "```bash",
        "python -m industrial_ai.stage_eval_inputs \\",
        "  --valid-input /path/to/official/eval_input_valid.csv \\",
        "  --anomaly-input /path/to/official/eval_input_anomaly.csv",
        "python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle",
        "```",
        "",
        "## Launch Commands",
        "",
        "Use the generated `artifacts/leonardo_launch_commands.sh` as the authoritative source.",
        "",
        "```bash",
        *launch_commands,
        "```",
        "",
        "## Post-Run Verification",
        "",
        "```bash",
        *verification_commands,
        "```",
        "",
        "## Copy Back",
        "",
        "Prefer copying back `artifacts/leonardo_return_packet.zip`, its `.sha256`, and "
        "`leonardo_return_packet_manifest.json`; verify it locally with "
        "`python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip "
        "--require-final-leonardo-objective`. If copying the directory instead, copy back "
        "`artifacts/submission_package/` including `track1_submission.zip`, "
        "`track1_submission.zip.sha256`, and `package_manifest.json`, then rerun the "
        "returned-package verification command above locally.",
        "",
    ])
    warnings = [str(item) for item in _as_list(payload.get("warnings"))]
    if warnings:
        lines.extend(["## Current Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_transfer_packet(
    packet_path: Path,
    payload: dict[str, object],
    checklist_path: Path,
    handoff_path: Path,
    readiness_path: Path,
    launch_commands_path: Path,
    selftest_path: Path,
) -> dict[str, object]:
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, str]] = []
    seen: set[str] = set()
    candidate_paths = [
        *[Path(str(item)) for item in _as_list(payload.get("upload_files"))],
        handoff_path,
        checklist_path,
        readiness_path,
        launch_commands_path,
        selftest_path,
    ]
    for path in candidate_paths:
        if not path.exists() or not path.is_file():
            continue
        arcname = path.name
        if arcname in seen:
            continue
        seen.add(arcname)
        entries.append((path, arcname))
    entry_rows = [
        {
            "path": arcname,
            "source": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path, arcname in entries
    ]
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "bundle_sha256": str(payload.get("bundle_sha256", "") or ""),
        "handoff_passed": payload.get("passed") is True,
        "entry_count": len(entry_rows),
        "entries": entry_rows,
        "unpack_commands": [
            f"sha256sum -c {packet_path.name}.sha256",
            f"unzip -o {packet_path.name}",
            "sha256sum -c leonardo_source_bundle.zip.sha256",
            "unzip -o leonardo_source_bundle.zip",
            f"python -m industrial_ai.leonardo_handoff --verify-transfer-packet {packet_path.name} --verify-fresh-unpack",
            "python -m industrial_ai.leonardo_bundle --verify-root . --manifest leonardo_source_bundle_manifest.json",
        ],
    }
    tmp_path = packet_path.with_suffix(packet_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname)
        zf.writestr("leonardo_transfer_packet_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if packet_path.exists():
        packet_path.unlink()
    tmp_path.replace(packet_path)
    packet_hash = file_sha256(packet_path)
    manifest["packet"] = {
        "path": str(packet_path),
        "bytes": packet_path.stat().st_size,
        "sha256": packet_hash,
        "sidecar": str(packet_path.with_suffix(packet_path.suffix + ".sha256")),
        "manifest": str(packet_path.with_name(packet_path.stem + "_manifest.json")),
    }
    packet_path.with_suffix(packet_path.suffix + ".sha256").write_text(
        f"{packet_hash}  {packet_path.name}\n",
        encoding="utf-8",
    )
    packet_path.with_name(packet_path.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _read_packet_sidecar_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    parts = text.split()
    return parts[0] if parts else ""


def _unsafe_zip_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not normalized:
        return "empty ZIP entry name"
    if normalized.startswith("/") or PurePosixPath(normalized).is_absolute():
        return f"absolute ZIP entry path: {name}"
    if any(part in ("", ".", "..") for part in parts):
        return f"unsafe ZIP entry path: {name}"
    return ""


def _safe_extract_all(zip_path: Path, destination: Path) -> list[str]:
    failures: list[str] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                unsafe = _unsafe_zip_name(name)
                if unsafe:
                    failures.append(unsafe)
                    continue
                target = destination / name.replace("\\", "/")
                target.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.write_bytes(zf.read(name))
    except (OSError, zipfile.BadZipFile) as exc:
        failures.append(f"Could not extract ZIP {zip_path}: {exc}")
    return failures


def _fresh_unpack_failures(packet_path: Path) -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="leonardo_transfer_unpack_") as temp_dir:
        root = Path(temp_dir)
        failures.extend(_safe_extract_all(packet_path, root))
        source_bundle = root / "leonardo_source_bundle.zip"
        source_sidecar = root / "leonardo_source_bundle.zip.sha256"
        source_manifest = root / "leonardo_source_bundle_manifest.json"
        if failures:
            return failures
        if not source_bundle.exists():
            return ["Fresh transfer-packet unpack did not produce leonardo_source_bundle.zip"]
        failures.extend(_safe_extract_all(source_bundle, root))
        if failures:
            return failures
        failures.extend(verify_bundle(source_manifest, root, source_bundle, source_sidecar))
    return failures


def verify_transfer_packet(
    packet_path: Path,
    sidecar_path: Path | None = None,
    manifest_path: Path | None = None,
    verify_fresh_unpack: bool = False,
) -> list[str]:
    failures: list[str] = []
    if sidecar_path is None:
        sidecar_path = packet_path.with_suffix(packet_path.suffix + ".sha256")
    if manifest_path is None:
        manifest_path = packet_path.with_name(packet_path.stem + "_manifest.json")
    if not packet_path.exists():
        return [f"Missing transfer packet ZIP: {packet_path}"]
    if not packet_path.is_file() or packet_path.stat().st_size <= 0:
        failures.append(f"Empty transfer packet ZIP: {packet_path}")
    packet_hash = file_sha256(packet_path) if packet_path.exists() and packet_path.is_file() else ""
    if not sidecar_path.exists():
        failures.append(f"Missing transfer packet checksum sidecar: {sidecar_path}")
    else:
        try:
            sidecar_hash = _read_packet_sidecar_hash(sidecar_path)
        except OSError as exc:
            failures.append(f"Transfer packet checksum sidecar is not readable: {sidecar_path} ({exc})")
            sidecar_hash = ""
        if sidecar_hash and packet_hash and sidecar_hash != packet_hash:
            failures.append(f"Transfer packet sidecar hash {sidecar_hash} does not match ZIP hash {packet_hash}")
    external_manifest: dict[str, object] = {}
    if not manifest_path.exists():
        failures.append(f"Missing transfer packet manifest: {manifest_path}")
    else:
        try:
            external_manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Transfer packet manifest is not readable JSON: {manifest_path} ({exc})")
    packet_info = external_manifest.get("packet", {}) if isinstance(external_manifest, dict) else {}
    if isinstance(packet_info, dict):
        manifest_hash = str(packet_info.get("sha256", "") or "")
        if manifest_hash and packet_hash and manifest_hash != packet_hash:
            failures.append(f"Transfer packet manifest hash {manifest_hash} does not match ZIP hash {packet_hash}")

    try:
        with zipfile.ZipFile(packet_path, "r") as zf:
            names = set(zf.namelist())
            for name in names:
                unsafe = _unsafe_zip_name(name)
                if unsafe:
                    failures.append(f"Transfer packet ZIP has {unsafe}")
            if "leonardo_transfer_packet_manifest.json" not in names:
                failures.append("Transfer packet ZIP missing embedded manifest")
                embedded_manifest: dict[str, object] = {}
            else:
                embedded_manifest = json.loads(
                    zf.read("leonardo_transfer_packet_manifest.json").decode("utf-8-sig")
                )
            entries = embedded_manifest.get("entries", []) if isinstance(embedded_manifest, dict) else []
            if not isinstance(entries, list) or not entries:
                failures.append("Transfer packet embedded manifest has no entries")
                entries = []
            expected_names = {"leonardo_transfer_packet_manifest.json"}
            required_names = {
                "leonardo_source_bundle.zip",
                "leonardo_source_bundle.zip.sha256",
                "leonardo_source_bundle_manifest.json",
                "leonardo_handoff.json",
                "leonardo_handoff_checklist.md",
                "leonardo_readiness.json",
                "leonardo_launch_commands.sh",
                "source_bundle_proof_selftest.json",
            }
            for item in entries:
                if not isinstance(item, dict):
                    failures.append("Transfer packet embedded manifest contains a non-object entry")
                    continue
                rel_path = str(item.get("path", "") or "").replace("\\", "/")
                if not rel_path:
                    failures.append("Transfer packet embedded manifest has an entry without path")
                    continue
                unsafe = _unsafe_zip_name(rel_path)
                if unsafe:
                    failures.append(f"Transfer packet embedded manifest has {unsafe}")
                    continue
                expected_names.add(rel_path)
                if rel_path not in names:
                    failures.append(f"Transfer packet ZIP missing manifest entry: {rel_path}")
                    continue
                data = zf.read(rel_path)
                expected_bytes = int(item.get("bytes", -1))
                if expected_bytes >= 0 and len(data) != expected_bytes:
                    failures.append(
                        f"Transfer packet entry size mismatch for {rel_path}: {len(data)} != {expected_bytes}"
                    )
                expected_hash = str(item.get("sha256", "") or "")
                if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                    failures.append(f"Transfer packet entry hash mismatch for {rel_path}")
            missing_required = sorted(required_names - names)
            if missing_required:
                failures.append("Transfer packet ZIP missing required handoff files: " + ", ".join(missing_required))
            unexpected = sorted(names - expected_names)
            if unexpected:
                failures.append("Transfer packet ZIP has unexpected entries: " + ", ".join(unexpected))
            bundle_rows = [
                item for item in entries
                if isinstance(item, dict) and item.get("path") == "leonardo_source_bundle.zip"
            ]
            embedded_bundle_sha = str(embedded_manifest.get("bundle_sha256", "") or "")
            if bundle_rows and embedded_bundle_sha and bundle_rows[0].get("sha256") != embedded_bundle_sha:
                failures.append(
                    "Transfer packet embedded bundle SHA does not match leonardo_source_bundle.zip entry hash"
                )
            if external_manifest and embedded_manifest:
                external_entries = external_manifest.get("entries", [])
                if isinstance(external_entries, list) and external_entries != entries:
                    failures.append("Transfer packet external manifest entries do not match embedded manifest entries")
                external_bundle_sha = str(external_manifest.get("bundle_sha256", "") or "")
                if external_bundle_sha and embedded_bundle_sha and external_bundle_sha != embedded_bundle_sha:
                    failures.append("Transfer packet external and embedded bundle SHA values differ")
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError) as exc:
        failures.append(f"Transfer packet ZIP is not readable: {packet_path} ({exc})")
    if verify_fresh_unpack and not failures:
        failures.extend(_fresh_unpack_failures(packet_path))
    return failures


def audit_handoff(
    bundle_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    launch_commands_path: Path,
    selftest_path: Path,
    require_source_bundle: bool,
) -> dict[str, object]:
    sidecar_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    failures: list[str] = []
    warnings: list[str] = []

    for path, label in (
        (bundle_path, "source bundle ZIP"),
        (sidecar_path, "source bundle sidecar"),
        (manifest_path, "source bundle manifest"),
        (readiness_path, "Leonardo readiness JSON"),
        (launch_commands_path, "Leonardo launch command script"),
        (selftest_path, "source-bundle proof self-test JSON"),
    ):
        if not path.exists():
            failures.append(f"Missing {label}: {path}")
        elif path.is_file() and path.stat().st_size <= 0:
            failures.append(f"Empty {label}: {path}")

    bundle_sha256 = file_sha256(bundle_path) if bundle_path.exists() and bundle_path.is_file() else ""
    sidecar_sha256 = ""
    if sidecar_path.exists():
        try:
            sidecar_sha256 = _read_sidecar_hash(sidecar_path)
        except OSError as exc:
            failures.append(f"Source bundle sidecar is not readable: {sidecar_path} ({exc})")
    if sidecar_sha256 and bundle_sha256 and sidecar_sha256 != bundle_sha256:
        failures.append(f"Source bundle sidecar hash {sidecar_sha256} does not match ZIP hash {bundle_sha256}")

    bundle_failures = verify_bundle(
        manifest_path,
        PROJECT_ROOT,
        bundle_path if bundle_path.exists() else None,
        sidecar_path if sidecar_path.exists() else None,
    )
    failures.extend(bundle_failures)

    readiness: dict[str, object] = {}
    if readiness_path.exists():
        try:
            readiness = _read_json(readiness_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Leonardo readiness is not readable JSON: {readiness_path} ({exc})")
    source_bundle = readiness.get("source_bundle", {}) if readiness else {}
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    readiness_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
    if readiness:
        if readiness.get("passed") is not True:
            failures.append("Leonardo readiness did not pass")
        if readiness.get("defer_eval_staging") is True:
            warnings.append(
                "Leonardo readiness deferred eval staging; upload handoff may proceed, "
                "but official eval CSVs must be staged and readiness rerun on Leonardo before jobs start"
            )
            if "data/eval/eval_input_valid.csv" not in "\n".join(_handoff_commands(readiness)):
                failures.append("Deferred-eval handoff commands do not point at data/eval/eval_input_valid.csv")
            if "data/eval/eval_input_anomaly.csv" not in "\n".join(_handoff_commands(readiness)):
                failures.append("Deferred-eval handoff commands do not point at data/eval/eval_input_anomaly.csv")
        if readiness.get("require_eval") is True and "REQUIRE_EVAL=1" not in "\n".join(_handoff_commands(readiness)):
            failures.append("Leonardo readiness handoff commands missing required eval export")
        batch_size = int(readiness.get("batch_size", 0) or 0)
        command_text = "\n".join(_handoff_commands(readiness))
        verification_commands = "\n".join(str(command) for command in _as_list(readiness.get("verification_commands")))
        if batch_size > 0:
            if f"BATCH_SIZE={batch_size}" not in command_text:
                failures.append(f"Leonardo readiness handoff commands missing batch-size export: BATCH_SIZE={batch_size}")
            if f"--required-batch-size {batch_size}" not in verification_commands:
                failures.append(
                    f"Leonardo readiness verification commands missing batch-size proof: --required-batch-size {batch_size}"
                )
        if require_source_bundle and readiness.get("require_source_bundle") is not True:
            failures.append("Leonardo readiness did not require source-bundle proof")
        if require_source_bundle and source_bundle.get("verified") is not True:
            failures.append("Leonardo readiness source_bundle.verified is not true")
        if bundle_sha256 and readiness_bundle_hash and readiness_bundle_hash != bundle_sha256:
            failures.append(
                f"Leonardo readiness source_bundle hash {readiness_bundle_hash} does not match ZIP hash {bundle_sha256}"
            )
        if source_bundle.get("failures"):
            failures.append(
                "Leonardo readiness source_bundle recorded failures: "
                + "; ".join(map(str, _as_list(source_bundle.get("failures"))))
            )

    launch_text = ""
    if launch_commands_path.exists():
        try:
            launch_text = launch_commands_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            failures.append(f"Leonardo launch command script is not readable: {launch_commands_path} ({exc})")
    if readiness:
        for command in _handoff_commands(readiness):
            if launch_text and str(command) not in launch_text:
                failures.append(f"Launch command script missing recorded readiness command: {command}")
    if readiness and readiness.get("require_eval") is True and launch_text and "REQUIRE_EVAL=1" not in launch_text:
        failures.append("Launch command script missing required eval export: REQUIRE_EVAL=1")
    if require_source_bundle:
        for needle, label in {
            "REQUIRE_SOURCE_BUNDLE=1": "source-bundle export",
            "source_bundle_proof_selftest": "source-bundle proof self-test",
            "verify_returned_package": "returned-package verification",
            "run_evidence_report": "objective evidence report",
        }.items():
            if launch_text and needle not in launch_text:
                failures.append(f"Launch command script missing {label}: {needle}")

    selftest: dict[str, object] = {}
    if selftest_path.exists():
        try:
            selftest = _read_json(selftest_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Source-bundle proof self-test is not readable JSON: {selftest_path} ({exc})")
    if selftest:
        if selftest.get("passed") is not True:
            failures.append("Source-bundle proof self-test did not pass")
        if selftest.get("failures"):
            failures.append(
                "Source-bundle proof self-test recorded failures: "
                + "; ".join(map(str, _as_list(selftest.get("failures"))))
            )
    elif require_source_bundle:
        failures.append("Missing readable source-bundle proof self-test payload")

    upload_files = [bundle_path.name, sidecar_path.name, manifest_path.name]
    if readiness:
        recorded_uploads = set()
        if isinstance(source_bundle.get("handoff_upload_files"), list):
            recorded_uploads = {str(item) for item in source_bundle["handoff_upload_files"]}
        missing_uploads = sorted(set(upload_files) - recorded_uploads)
        if require_source_bundle and missing_uploads:
            failures.append("Readiness source-bundle handoff upload list is missing: " + ", ".join(missing_uploads))
        selftest_commands = "\n".join(str(item) for item in _as_list(source_bundle.get("handoff_selftest_commands")))
        if require_source_bundle and "source_bundle_proof_selftest" not in selftest_commands:
            failures.append("Readiness source-bundle handoff does not record source-bundle self-test command")
        strict_commands = "\n".join(str(item) for item in _as_list(source_bundle.get("handoff_readiness_commands")))
        if require_source_bundle:
            if "--require-source-bundle" not in strict_commands:
                failures.append("Readiness source-bundle handoff does not record strict source-bundle readiness command")
            if "--require-eval" not in strict_commands:
                failures.append("Readiness source-bundle handoff does not record strict eval readiness command")
            if "--defer-eval-staging" in strict_commands:
                failures.append("Readiness source-bundle handoff strict readiness command still defers eval staging")
        if readiness.get("defer_eval_staging") is True:
            deferred_commands = "\n".join(
                str(item) for item in _as_list(source_bundle.get("handoff_deferred_eval_readiness_commands"))
            )
            if "--defer-eval-staging" not in deferred_commands:
                failures.append("Readiness source-bundle handoff does not record deferred-eval readiness command")
            if "--require-eval" not in deferred_commands:
                failures.append("Readiness source-bundle handoff deferred readiness command does not require eval")
        audit_commands = "\n".join(str(item) for item in _as_list(source_bundle.get("handoff_audit_commands")))
        if require_source_bundle and ("leonardo_handoff" not in audit_commands or "--require-source-bundle" not in audit_commands):
            failures.append("Readiness source-bundle handoff does not record source-bundle handoff audit command")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "project_root": str(PROJECT_ROOT),
        "upload_files": [str(path) for path in (bundle_path, sidecar_path, manifest_path)],
        "bundle_sha256": bundle_sha256,
        "readiness_bundle_sha256": readiness_bundle_hash,
        "require_source_bundle": require_source_bundle,
        "readiness_path": str(readiness_path),
        "launch_commands_path": str(launch_commands_path),
        "selftest_path": str(selftest_path),
        "handoff_commands": _handoff_commands(readiness) if readiness else [],
        "warnings": warnings,
        "failures": failures,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the local Leonardo upload handoff artifacts.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BUNDLE_MANIFEST_PATH)
    parser.add_argument("--readiness", type=Path, default=PROJECT_ROOT / "artifacts" / "leonardo_readiness.json")
    parser.add_argument(
        "--launch-commands",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "leonardo_launch_commands.sh",
    )
    parser.add_argument(
        "--selftest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "source_bundle_proof_selftest.json",
    )
    parser.add_argument("--require-source-bundle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "leonardo_handoff.json")
    parser.add_argument(
        "--checklist-out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "leonardo_handoff_checklist.md",
        help="Write a human-readable upload, launch, and returned-package verification checklist.",
    )
    parser.add_argument(
        "--packet-out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "leonardo_transfer_packet.zip",
        help="Write a single transfer ZIP containing upload files plus handoff/checklist evidence.",
    )
    parser.add_argument(
        "--verify-transfer-packet",
        type=Path,
        default=None,
        help="Verify a Leonardo transfer packet ZIP and exit.",
    )
    parser.add_argument("--packet-sidecar", type=Path, default=None)
    parser.add_argument("--packet-manifest", type=Path, default=None)
    parser.add_argument(
        "--verify-fresh-unpack",
        action="store_true",
        help="Also extract the packet and nested source bundle into a temp directory and verify the resulting tree.",
    )
    args = parser.parse_args()
    if args.verify_transfer_packet is not None:
        failures = verify_transfer_packet(
            args.verify_transfer_packet,
            args.packet_sidecar,
            args.packet_manifest,
            args.verify_fresh_unpack,
        )
        if failures:
            print("Leonardo transfer packet verification failed:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(2)
        print("Leonardo transfer packet verification passed")
        return
    args.bundle, args.manifest = _effective_source_bundle_paths(args.bundle, args.manifest)

    payload = audit_handoff(
        args.bundle,
        args.manifest,
        args.readiness,
        args.launch_commands,
        args.selftest,
        args.require_source_bundle,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.checklist_out:
        _write_checklist(args.checklist_out, payload)
        print(f"Wrote {args.checklist_out}")
    if args.packet_out:
        _write_transfer_packet(
            args.packet_out,
            payload,
            args.checklist_out,
            args.out,
            args.readiness,
            args.launch_commands,
            args.selftest,
        )
        print(f"Wrote {args.packet_out}")
        print(f"Wrote {args.packet_out.with_suffix(args.packet_out.suffix + '.sha256')}")
        print(f"Wrote {args.packet_out.with_name(args.packet_out.stem + '_manifest.json')}")
    if payload["warnings"]:
        print("Leonardo handoff warnings:")
        for warning in payload["warnings"]:
            print(f"- {warning}")
    if payload["failures"]:
        print("Leonardo handoff audit failed:")
        for failure in payload["failures"]:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Leonardo handoff audit passed")
    print(f"Bundle SHA256: {payload['bundle_sha256']}")


if __name__ == "__main__":
    main()
