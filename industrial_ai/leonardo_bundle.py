from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .hashing import file_sha256
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "leonardo_source_bundle.zip"
DEFAULT_BUNDLE_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "leonardo_source_bundle_manifest.json"
CORE_FILES = [
    ".env.example",
    ".gitignore",
    "LEONARDO_RUNBOOK.md",
    "LICENSE",
    "README.md",
    "REPORT.md",
    "SLIDES.md",
    "SLIDES.pdf",
    "SUBMISSION_CHECKLIST.md",
    "VIDEO_SCRIPT.md",
    "requirements.txt",
    "artifacts/slurm/.gitkeep",
    "data/eval/.gitkeep",
]
SOURCE_DIRS = [
    "industrial_ai",
    "scripts",
]
EXCLUDE_PATTERNS = [
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.log",
    ".env",
    ".venv/*",
    "artifacts/*",
    "checkpoints/*",
    "data/dev/*",
    "data/generated/*",
    "submissions/*",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _is_excluded(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in EXCLUDE_PATTERNS)


def _iter_source_files(include_eval: bool) -> list[Path]:
    files: dict[str, Path] = {}
    for rel_name in CORE_FILES:
        path = PROJECT_ROOT / rel_name
        if path.exists() and path.is_file():
            files[_rel(path)] = path
    for directory in SOURCE_DIRS:
        root = PROJECT_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                rel_path = _rel(path)
                if not _is_excluded(rel_path):
                    files[rel_path] = path
    for path in DEFAULT_DATA_DIR.glob("*.csv"):
        if path.is_file():
            files[_rel(path)] = path
    generator = DEFAULT_DATA_DIR / "generate_sequences.py"
    if generator.exists() and generator.is_file():
        files[_rel(generator)] = generator
    if include_eval:
        for path in (PROJECT_ROOT / "data" / "eval").glob("*.csv"):
            if path.is_file():
                files[_rel(path)] = path
        eval_manifest = PROJECT_ROOT / "artifacts" / "eval_staging_manifest.json"
        if eval_manifest.exists() and eval_manifest.is_file():
            files[_rel(eval_manifest)] = eval_manifest
    return [files[key] for key in sorted(files)]


def _write_sidecar(path: Path, bundle_hash: str, manifest: dict[str, object]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{bundle_hash}  {path.name}\n", encoding="utf-8")
    manifest_path = path.with_name(path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _handoff_metadata(out: Path) -> dict[str, object]:
    manifest_name = out.with_name(out.stem + "_manifest.json").name
    sidecar_name = out.with_suffix(out.suffix + ".sha256").name
    bundle_name = out.name
    return {
        "upload_files": [
            bundle_name,
            sidecar_name,
            manifest_name,
        ],
        "unpack_command": f"unzip -o {bundle_name}",
        "pre_unpack_verify_commands": [
            f"sha256sum -c {sidecar_name}",
        ],
        "verify_commands": [
            f"sha256sum -c {sidecar_name}",
            f"unzip -o {bundle_name}",
            f"python -m industrial_ai.leonardo_bundle --verify-bundle {bundle_name} --sidecar {sidecar_name}",
            f"python -m industrial_ai.leonardo_bundle --verify-root . --manifest {manifest_name}",
        ],
        "readiness_commands": [
            "python -m industrial_ai.leonardo_readiness --count-profile standard --require-eval --require-source-bundle",
            "python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle",
        ],
        "deferred_eval_readiness_commands": [
            "python -m industrial_ai.leonardo_readiness --count-profile standard --require-eval --defer-eval-staging --require-source-bundle",
            "python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --defer-eval-staging --require-source-bundle",
        ],
        "selftest_commands": [
            "python -m industrial_ai.source_bundle_proof_selftest",
        ],
        "handoff_audit_commands": [
            "python -m industrial_ai.leonardo_handoff --require-source-bundle",
        ],
    }


def create_bundle(out: Path, include_eval: bool) -> dict[str, object]:
    files = _iter_source_files(include_eval)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(out.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()
    manifest_files = []
    for path in files:
        rel_path = _rel(path)
        manifest_files.append({
            "path": rel_path,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "include_eval": include_eval,
        "file_count": len(manifest_files),
        "files": manifest_files,
        "handoff": _handoff_metadata(out),
    }
    with zipfile.ZipFile(tmp_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in manifest_files:
            rel_path = str(item["path"])
            zf.write(PROJECT_ROOT / rel_path, rel_path)
        zf.writestr("leonardo_source_bundle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if out.exists():
        out.unlink()
    tmp_out.replace(out)
    bundle_hash = file_sha256(out)
    manifest["bundle"] = {
        "path": str(out),
        "bytes": out.stat().st_size,
        "sha256": bundle_hash,
        "sidecar": str(out.with_suffix(out.suffix + ".sha256")),
    }
    _write_sidecar(out, bundle_hash, manifest)
    return manifest


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_sidecar_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    return text.split()[0] if text else ""


def _read_embedded_manifest(bundle_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle_path, "r") as zf:
        return json.loads(zf.read("leonardo_source_bundle_manifest.json").decode("utf-8-sig"))


def _verify_bundle_zip(bundle_path: Path, sidecar_path: Path | None, manifest: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if not bundle_path.exists():
        return [f"Missing bundle ZIP: {bundle_path}"]
    expected_hash = ""
    bundle_info = manifest.get("bundle", {})
    if isinstance(bundle_info, dict):
        expected_hash = str(bundle_info.get("sha256", "") or "")
    if sidecar_path is not None:
        if not sidecar_path.exists():
            failures.append(f"Missing bundle checksum sidecar: {sidecar_path}")
        else:
            sidecar_hash = _read_sidecar_hash(sidecar_path)
            if expected_hash and sidecar_hash != expected_hash:
                failures.append(
                    f"Bundle sidecar hash is {sidecar_hash}; expected manifest hash {expected_hash}"
                )
            expected_hash = expected_hash or sidecar_hash
    actual_hash = file_sha256(bundle_path)
    if expected_hash and actual_hash != expected_hash:
        failures.append(f"Bundle ZIP hash is {actual_hash}; expected {expected_hash}")
    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = set(zf.namelist())
            expected_names = {"leonardo_source_bundle_manifest.json"}
            for item in manifest.get("files", []):
                if isinstance(item, dict):
                    rel_path = str(item.get("path", "") or "").replace("\\", "/")
                    if rel_path:
                        expected_names.add(rel_path)
            unexpected_names = sorted(names - expected_names)
            if unexpected_names:
                failures.append("Bundle ZIP has unexpected entries: " + ", ".join(unexpected_names))
            for item in manifest.get("files", []):
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path", "") or "").replace("\\", "/")
                if not rel_path:
                    failures.append("Bundle manifest has a file entry without path")
                    continue
                if rel_path not in names:
                    failures.append(f"Bundle ZIP missing manifest file: {rel_path}")
                    continue
                data = zf.read(rel_path)
                expected_bytes = int(item.get("bytes", -1))
                if expected_bytes >= 0 and len(data) != expected_bytes:
                    failures.append(
                        f"Bundle ZIP entry size mismatch for {rel_path}: {len(data)} != {expected_bytes}"
                    )
                expected_entry_hash = str(item.get("sha256", "") or "")
                if expected_entry_hash:
                    actual_entry_hash = hashlib.sha256(data).hexdigest()
                    if actual_entry_hash != expected_entry_hash:
                        failures.append(
                            f"Bundle ZIP entry hash mismatch for {rel_path}: "
                            f"{actual_entry_hash} != {expected_entry_hash}"
                        )
            if "leonardo_source_bundle_manifest.json" not in names:
                failures.append("Bundle ZIP missing leonardo_source_bundle_manifest.json")
    except zipfile.BadZipFile as exc:
        failures.append(f"Bundle ZIP is not readable: {bundle_path} ({exc})")
    return failures


def _verify_unpacked_tree(root: Path, manifest: dict[str, object]) -> list[str]:
    failures: list[str] = []
    root = root.resolve()
    manifest_paths = {
        str(item.get("path", "") or "").replace("\\", "/")
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            failures.append("Bundle manifest has a non-object file entry")
            continue
        rel_path = str(item.get("path", "") or "").replace("\\", "/")
        if not rel_path:
            failures.append("Bundle manifest has a file entry without path")
            continue
        path = root / rel_path
        if not path.exists():
            failures.append(f"Unpacked bundle missing file: {rel_path}")
            continue
        if not path.is_file():
            failures.append(f"Unpacked bundle path is not a file: {rel_path}")
            continue
        expected_bytes = int(item.get("bytes", -1))
        if expected_bytes >= 0 and path.stat().st_size != expected_bytes:
            failures.append(
                f"Unpacked bundle file size mismatch for {rel_path}: "
                f"{path.stat().st_size} != {expected_bytes}"
            )
        expected_hash = str(item.get("sha256", "") or "")
        if expected_hash:
            actual_hash = file_sha256(path)
            if actual_hash != expected_hash:
                failures.append(f"Unpacked bundle hash mismatch for {rel_path}: {actual_hash} != {expected_hash}")
    for directory in SOURCE_DIRS:
        source_root = root / directory
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(root)).replace("\\", "/")
            if _is_excluded(rel_path):
                continue
            if rel_path not in manifest_paths:
                failures.append(f"Unpacked source tree has unexpected executable/source file: {rel_path}")
    return failures


def verify_bundle(
    manifest_path: Path,
    verify_root: Path | None,
    bundle_path: Path | None,
    sidecar_path: Path | None,
) -> list[str]:
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            return [f"Bundle manifest is not readable JSON: {manifest_path} ({exc})"]
    elif bundle_path is not None:
        try:
            manifest = _read_embedded_manifest(bundle_path)
        except KeyError:
            return [f"Bundle ZIP missing embedded manifest: {bundle_path}!leonardo_source_bundle_manifest.json"]
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [f"Bundle embedded manifest is not readable: {bundle_path} ({exc})"]
    else:
        return [f"Missing bundle manifest: {manifest_path}"]
    failures: list[str] = []
    if verify_root is not None:
        failures.extend(_verify_unpacked_tree(verify_root, manifest))
    if bundle_path is not None:
        failures.extend(_verify_bundle_zip(bundle_path, sidecar_path, manifest))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible source bundle for staging on Leonardo.")
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Include data/eval/*.csv if official eval files are already staged locally.",
    )
    parser.add_argument(
        "--verify-root",
        type=Path,
        default=None,
        help="Verify an unpacked source tree against a bundle manifest instead of creating a bundle.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BUNDLE_MANIFEST_PATH,
        help=(
            "Bundle manifest to verify. For --verify-bundle, the embedded "
            "leonardo_source_bundle_manifest.json is used when this file is absent."
        ),
    )
    parser.add_argument(
        "--verify-bundle",
        type=Path,
        default=None,
        help="Verify a bundle ZIP hash and entries against the bundle manifest.",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Optional .sha256 sidecar to use when verifying a bundle ZIP.",
    )
    args = parser.parse_args()
    if args.verify_root is not None or args.verify_bundle is not None:
        failures = verify_bundle(args.manifest, args.verify_root, args.verify_bundle, args.sidecar)
        if failures:
            print("Leonardo source bundle verification failed:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(2)
        print("Leonardo source bundle verification passed")
        return
    manifest = create_bundle(args.out, args.include_eval)
    bundle = manifest["bundle"]
    print(f"Wrote {bundle['path']}")
    print(f"Wrote {bundle['sidecar']}")
    print(f"Wrote {args.out.with_name(args.out.stem + '_manifest.json')}")
    print(f"Files: {manifest['file_count']}")
    print(f"SHA256: {bundle['sha256']}")


if __name__ == "__main__":
    main()
