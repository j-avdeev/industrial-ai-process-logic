from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .hashing import file_sha256
from .paths import PROJECT_ROOT
from .verify_package import verify_package


DEFAULT_PACKET = PROJECT_ROOT / "artifacts" / "leonardo_return_packet.zip"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _collect_entries(artifacts_dir: Path, package_dir: Path) -> list[tuple[Path, str]]:
    candidates = [
        package_dir / "track1_submission.zip",
        package_dir / "track1_submission.zip.sha256",
        package_dir / "package_manifest.json",
        artifacts_dir / "returned_package_verification.json",
        artifacts_dir / "final_audit_summary.json",
        artifacts_dir / "run_evidence_report.json",
        artifacts_dir / "run_evidence_report.md",
    ]
    entries: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        if path.parent == package_dir:
            arcname = f"submission_package/{path.name}"
        else:
            arcname = path.name
        if arcname in seen:
            continue
        seen.add(arcname)
        entries.append((path, arcname))
    return entries


def create_return_packet(
    artifacts_dir: Path,
    package_dir: Path,
    packet_path: Path,
) -> dict[str, object]:
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    entries = _collect_entries(artifacts_dir, package_dir)
    entry_rows = [
        {
            "path": arcname,
            "source": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path, arcname in entries
    ]
    returned_summary = (
        _read_json(artifacts_dir / "returned_package_verification.json")
        if (artifacts_dir / "returned_package_verification.json").exists()
        else {}
    )
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "artifacts_dir": str(artifacts_dir),
        "package_dir": str(package_dir),
        "entry_count": len(entry_rows),
        "entries": entry_rows,
        "returned_package_verification": {
            "passed": returned_summary.get("passed"),
            "objective_ready": returned_summary.get("objective_ready"),
            "objective_scope": returned_summary.get("objective_scope"),
            "final_leonardo_objective_ready": returned_summary.get("final_leonardo_objective_ready"),
            "package_zip_sha256": returned_summary.get("package_zip_sha256"),
            "package_sidecar_sha256": returned_summary.get("package_sidecar_sha256"),
            "package_manifest_sha256": returned_summary.get("package_manifest_sha256"),
        },
    }
    tmp_path = packet_path.with_suffix(packet_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname)
        zf.writestr("leonardo_return_packet_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
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


def _read_sidecar_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    return text.split()[0] if text else ""


def verify_return_packet(
    packet_path: Path,
    sidecar_path: Path | None = None,
    manifest_path: Path | None = None,
    require_final_leonardo_objective: bool = False,
    verify_package_payload: bool = True,
) -> list[str]:
    failures: list[str] = []
    if sidecar_path is None:
        sidecar_path = packet_path.with_suffix(packet_path.suffix + ".sha256")
    if manifest_path is None:
        manifest_path = packet_path.with_name(packet_path.stem + "_manifest.json")
    if not packet_path.exists():
        return [f"Missing return packet ZIP: {packet_path}"]
    packet_hash = file_sha256(packet_path)
    if not sidecar_path.exists():
        failures.append(f"Missing return packet checksum sidecar: {sidecar_path}")
    else:
        try:
            sidecar_hash = _read_sidecar_hash(sidecar_path)
        except OSError as exc:
            failures.append(f"Return packet checksum sidecar is not readable: {sidecar_path} ({exc})")
            sidecar_hash = ""
        if sidecar_hash and sidecar_hash != packet_hash:
            failures.append(f"Return packet sidecar hash {sidecar_hash} does not match ZIP hash {packet_hash}")
    external_manifest: dict[str, object] = {}
    if not manifest_path.exists():
        failures.append(f"Missing return packet manifest: {manifest_path}")
    else:
        try:
            external_manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Return packet manifest is not readable JSON: {manifest_path} ({exc})")
    packet_info = external_manifest.get("packet", {}) if isinstance(external_manifest, dict) else {}
    if isinstance(packet_info, dict):
        manifest_hash = str(packet_info.get("sha256", "") or "")
        if manifest_hash and manifest_hash != packet_hash:
            failures.append(f"Return packet manifest hash {manifest_hash} does not match ZIP hash {packet_hash}")
    try:
        with zipfile.ZipFile(packet_path, "r") as zf:
            names = set(zf.namelist())
            for name in names:
                unsafe = _unsafe_zip_name(name)
                if unsafe:
                    failures.append(f"Return packet ZIP has {unsafe}")
            if "leonardo_return_packet_manifest.json" not in names:
                failures.append("Return packet ZIP missing embedded manifest")
                embedded_manifest: dict[str, object] = {}
            else:
                embedded_manifest = json.loads(
                    zf.read("leonardo_return_packet_manifest.json").decode("utf-8-sig")
                )
            entries = embedded_manifest.get("entries", []) if isinstance(embedded_manifest, dict) else []
            if not isinstance(entries, list) or not entries:
                failures.append("Return packet embedded manifest has no entries")
                entries = []
            expected_names = {"leonardo_return_packet_manifest.json"}
            required_names = {
                "submission_package/track1_submission.zip",
                "submission_package/track1_submission.zip.sha256",
                "submission_package/package_manifest.json",
                "returned_package_verification.json",
                "final_audit_summary.json",
                "run_evidence_report.json",
            }
            for item in entries:
                if not isinstance(item, dict):
                    failures.append("Return packet embedded manifest contains a non-object entry")
                    continue
                rel_path = str(item.get("path", "") or "").replace("\\", "/")
                if not rel_path:
                    failures.append("Return packet embedded manifest has an entry without path")
                    continue
                unsafe = _unsafe_zip_name(rel_path)
                if unsafe:
                    failures.append(f"Return packet embedded manifest has {unsafe}")
                    continue
                expected_names.add(rel_path)
                if rel_path not in names:
                    failures.append(f"Return packet ZIP missing manifest entry: {rel_path}")
                    continue
                data = zf.read(rel_path)
                expected_bytes = int(item.get("bytes", -1))
                if expected_bytes >= 0 and len(data) != expected_bytes:
                    failures.append(f"Return packet entry size mismatch for {rel_path}: {len(data)} != {expected_bytes}")
                expected_hash = str(item.get("sha256", "") or "")
                if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                    failures.append(f"Return packet entry hash mismatch for {rel_path}")
            missing_required = sorted(required_names - names)
            if missing_required:
                failures.append("Return packet ZIP missing required returned-run files: " + ", ".join(missing_required))
            unexpected = sorted(names - expected_names)
            if unexpected:
                failures.append("Return packet ZIP has unexpected entries: " + ", ".join(unexpected))
            summary = embedded_manifest.get("returned_package_verification", {})
            if not isinstance(summary, dict):
                failures.append("Return packet embedded manifest has no returned_package_verification summary")
            else:
                if summary.get("passed") is not True:
                    failures.append("Return packet returned_package_verification.passed is not true")
                if summary.get("objective_ready") is not True:
                    failures.append("Return packet returned_package_verification.objective_ready is not true")
                if require_final_leonardo_objective and summary.get("final_leonardo_objective_ready") is not True:
                    failures.append(
                        "Return packet returned_package_verification.final_leonardo_objective_ready is not true"
                    )
            if external_manifest and embedded_manifest:
                external_entries = external_manifest.get("entries", [])
                if isinstance(external_entries, list) and external_entries != entries:
                    failures.append("Return packet external manifest entries do not match embedded manifest entries")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        failures.append(f"Return packet ZIP is not readable: {packet_path} ({exc})")
    if verify_package_payload and not failures:
        with tempfile.TemporaryDirectory(prefix="leonardo_return_packet_") as temp_dir:
            root = Path(temp_dir)
            extract_failures = _safe_extract_all(packet_path, root)
            if extract_failures:
                failures.extend(extract_failures)
            else:
                failures.extend(verify_package(root / "submission_package"))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify a Leonardo returned-artifacts packet.")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--package-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "submission_package")
    parser.add_argument("--out", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--verify", type=Path, default=None, help="Verify an existing return packet instead of creating one.")
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--require-final-leonardo-objective", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-package-payload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.verify is not None:
        failures = verify_return_packet(
            args.verify,
            args.sidecar,
            args.manifest,
            args.require_final_leonardo_objective,
            args.verify_package_payload,
        )
        if failures:
            print("Leonardo return packet verification failed:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(2)
        print("Leonardo return packet verification passed")
        return
    manifest = create_return_packet(args.artifacts_dir, args.package_dir, args.out)
    failures = verify_return_packet(
        args.out,
        require_final_leonardo_objective=args.require_final_leonardo_objective,
        verify_package_payload=args.verify_package_payload,
    )
    if failures:
        print("Leonardo return packet verification failed after create:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    packet = manifest["packet"]
    print(f"Wrote {packet['path']}")
    print(f"Wrote {packet['sidecar']}")
    print(f"Wrote {packet['manifest']}")
    print(f"Files: {manifest['entry_count']}")
    print(f"SHA256: {packet['sha256']}")


if __name__ == "__main__":
    main()
