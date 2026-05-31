from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .hashing import file_sha256
from .paths import DEFAULT_ARTIFACTS_DIR, PROJECT_ROOT
from .preflight import EXPECTED_EVAL_COLUMNS


DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_VALID_NAME = "eval_input_valid.csv"
DEFAULT_ANOMALY_NAME = "eval_input_anomaly.csv"


def _csv_status(label: str, path: Path) -> tuple[dict[str, object], list[str]]:
    expected_columns = EXPECTED_EVAL_COLUMNS[label]
    failures: list[str] = []
    status: dict[str, object] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": "",
        "rows": 0,
        "fieldnames": [],
        "required_columns": expected_columns,
        "missing_columns": expected_columns,
    }
    if not path.exists():
        return status, [f"Missing {label} eval CSV: {path}"]
    if not path.is_file():
        return status, [f"{label} eval input is not a file: {path}"]
    if not status["bytes"]:
        return status, [f"{label} eval CSV is empty: {path}"]
    try:
        status["sha256"] = file_sha256(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return status, [f"{label} eval CSV is not readable: {path} ({exc})"]

    missing_columns = [column for column in expected_columns if column not in fieldnames]
    status["fieldnames"] = fieldnames
    status["rows"] = rows
    status["missing_columns"] = missing_columns
    if not fieldnames:
        failures.append(f"{label} eval CSV has no header: {path}")
    elif missing_columns:
        failures.append(f"{label} eval CSV is missing columns: {', '.join(missing_columns)}")
    elif rows <= 0:
        failures.append(f"{label} eval CSV has no rows: {path}")
    return status, failures


def _copy_eval(src: Path, dst: Path, force: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if file_sha256(src) == file_sha256(dst):
            return
        if not force:
            raise FileExistsError(f"{dst} already exists with different content; pass --force to replace it")
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and stage official eval CSVs into data/eval.")
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--anomaly-input", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "eval_staging_manifest.json",
        help="Write a manifest with source/destination hashes and row counts.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing staged CSVs if their content differs.")
    args = parser.parse_args()

    failures: list[str] = []
    source_rows: list[dict[str, object]] = []
    for label, source in (("valid", args.valid_input), ("anomaly", args.anomaly_input)):
        status, status_failures = _csv_status(label, source)
        source_rows.append(status)
        failures.extend(status_failures)
    if failures:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "eval_dir": str(args.eval_dir),
            "sources": source_rows,
            "destinations": [],
            "failures": failures,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {args.out}")
        print("Eval staging failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)

    args.eval_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "valid": args.eval_dir / DEFAULT_VALID_NAME,
        "anomaly": args.eval_dir / DEFAULT_ANOMALY_NAME,
    }
    try:
        _copy_eval(args.valid_input, destinations["valid"], args.force)
        _copy_eval(args.anomaly_input, destinations["anomaly"], args.force)
    except OSError as exc:
        failures.append(str(exc))

    destination_rows: list[dict[str, object]] = []
    if not failures:
        for label, destination in destinations.items():
            status, status_failures = _csv_status(label, destination)
            destination_rows.append(status)
            failures.extend(status_failures)
            source = next(row for row in source_rows if row["label"] == label)
            if status.get("sha256") != source.get("sha256"):
                failures.append(
                    f"Staged {label} CSV hash does not match source: "
                    f"{status.get('sha256') or 'missing'} != {source.get('sha256') or 'missing'}"
                )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "eval_dir": str(args.eval_dir),
        "sources": source_rows,
        "destinations": destination_rows,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    if failures:
        print("Eval staging failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Eval staging passed")
    print(f"Valid: {destinations['valid']}")
    print(f"Anomaly: {destinations['anomaly']}")


if __name__ == "__main__":
    main()
