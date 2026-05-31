from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .hashing import file_sha256
from .paths import PROJECT_ROOT


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_arcname(path: Path, package_dir: Path) -> str:
    try:
        return str(path.relative_to(package_dir)).replace("\\", "/")
    except ValueError:
        parts = path.parts
        if "evidence" in parts:
            evidence_index = parts.index("evidence")
            return "/".join(parts[evidence_index:])
        return path.name.replace("\\", "/")


def _expected_zip_entries(package_dir: Path, manifest: dict[str, object], manifest_sha256: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        expected[path.name] = str(item.get("sha256", "") or "")
    expected["package_manifest.json"] = manifest_sha256
    for item in manifest.get("evidence", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        expected[_zip_arcname(path, package_dir)] = str(item.get("sha256", "") or "")
    return expected


def _check_unpacked_entries(package_dir: Path, manifest: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for section in ("files", "evidence"):
        for item in manifest.get(section, []):
            if not isinstance(item, dict):
                continue
            manifest_path = Path(str(item.get("path", "") or ""))
            local_path = package_dir / _zip_arcname(manifest_path, package_dir)
            if not local_path.exists():
                continue
            expected = str(item.get("sha256", "") or "")
            if not expected:
                failures.append(f"Package manifest has no hash for unpacked {section} file: {local_path}")
                continue
            actual = file_sha256(local_path)
            if actual != expected:
                failures.append(
                    f"Unpacked {section} file hash mismatch for {local_path}: expected {expected}, got {actual}"
                )
    return failures


def _check_sidecar(zip_path: Path, sidecar_path: Path) -> list[str]:
    if not sidecar_path.exists():
        return [f"Missing ZIP checksum sidecar: {sidecar_path}"]
    try:
        parts = sidecar_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        return [f"Could not read ZIP checksum sidecar: {sidecar_path} ({exc})"]
    if len(parts) < 2:
        return [f"Malformed ZIP checksum sidecar: {sidecar_path}"]
    expected_hash, expected_name = parts[0], parts[1]
    failures: list[str] = []
    if expected_name != zip_path.name:
        failures.append(f"ZIP checksum sidecar names {expected_name!r}; expected {zip_path.name!r}")
    actual_hash = file_sha256(zip_path) if zip_path.exists() else ""
    if actual_hash != expected_hash:
        failures.append(f"ZIP checksum mismatch: expected {expected_hash}, got {actual_hash}")
    return failures


def _read_manifest(package_dir: Path, zip_path: Path) -> tuple[dict[str, object], str, list[str]]:
    manifest_path = package_dir / "package_manifest.json"
    if manifest_path.exists():
        try:
            return _read_json(manifest_path), file_sha256(manifest_path), []
        except (OSError, json.JSONDecodeError) as exc:
            return {}, "", [f"Package manifest is not readable JSON: {manifest_path} ({exc})"]
    if not zip_path.exists():
        return {}, "", [f"Missing package manifest: {manifest_path}; package ZIP also missing: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                raw = zf.read("package_manifest.json")
            except KeyError:
                return {}, "", [f"Missing package manifest: {manifest_path} and ZIP entry package_manifest.json"]
    except zipfile.BadZipFile as exc:
        return {}, "", [f"Package ZIP is not readable while checking package_manifest.json: {zip_path} ({exc})"]
    try:
        return json.loads(raw.decode("utf-8-sig")), _sha256_bytes(raw), []
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, "", [f"Package ZIP manifest is not readable JSON: {zip_path}!package_manifest.json ({exc})"]


def verify_package(package_dir: Path) -> list[str]:
    failures: list[str] = []
    zip_path = package_dir / "track1_submission.zip"
    sidecar_path = package_dir / "track1_submission.zip.sha256"

    if not zip_path.exists():
        failures.append(f"Missing package ZIP: {zip_path}")
    if failures:
        return failures

    failures.extend(_check_sidecar(zip_path, sidecar_path))
    manifest, manifest_sha256, manifest_failures = _read_manifest(package_dir, zip_path)
    if manifest_failures:
        return [*failures, *manifest_failures]

    expected = _expected_zip_entries(package_dir, manifest, manifest_sha256)
    if not expected:
        failures.append("Package manifest has no expected ZIP entries")
    failures.extend(_check_unpacked_entries(package_dir, manifest))

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            for name in duplicates:
                failures.append(f"ZIP has duplicate entry: {name}")
            actual = set(names)
            expected_names = set(expected)
            for name in sorted(expected_names - actual):
                failures.append(f"ZIP missing expected entry: {name}")
            for name in sorted(actual - expected_names):
                failures.append(f"ZIP has unexpected entry: {name}")
            for name, digest in sorted(expected.items()):
                if name not in actual:
                    continue
                if not digest:
                    failures.append(f"Package manifest has no hash for ZIP entry: {name}")
                    continue
                actual_digest = _sha256_bytes(zf.read(name))
                if actual_digest != digest:
                    failures.append(f"ZIP entry hash mismatch for {name}: expected {digest}, got {actual_digest}")
    except zipfile.BadZipFile as exc:
        failures.append(f"Package ZIP is not readable: {zip_path} ({exc})")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a packaged Track 1 ZIP after transfer from Leonardo.")
    parser.add_argument("--package-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "submission_package")
    args = parser.parse_args()

    failures = verify_package(args.package_dir)
    if failures:
        print("Package verification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)

    print("Package verification passed")
    print(f"Package dir: {args.package_dir}")
    print(f"ZIP: {args.package_dir / 'track1_submission.zip'}")
    print(f"Checksum: {args.package_dir / 'track1_submission.zip.sha256'}")


if __name__ == "__main__":
    main()
