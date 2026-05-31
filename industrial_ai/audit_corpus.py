from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import FAMILY_FILES, SequenceRecord, corpus_fingerprint, infer_family_from_name, read_long_sequences, write_rows
from .hashing import file_sha256
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


def _summarize_records(
    path: Path,
    source: str,
    records: list[SequenceRecord],
    family: str | None = None,
) -> dict[str, object]:
    family_counts = Counter(record.family for record in records)
    lengths = [len(record.steps) for record in records]
    metadata = _read_generation_metadata(path) if source == "generated" else {}
    path_sha256 = file_sha256(path)
    return {
        "path": str(path),
        "source": source,
        "family": family or (next(iter(family_counts)) if len(family_counts) == 1 else infer_family_from_name(path.name)),
        "num_sequences": len(records),
        "num_steps": sum(lengths),
        "file_sha256": path_sha256,
        "content_sha256": corpus_fingerprint(records),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "metadata_path": str(path.with_suffix(path.suffix + ".metadata.json")) if source == "generated" else "",
        "metadata_present": bool(metadata),
        "metadata_status": metadata.get("status", ""),
        "metadata_family": metadata.get("family", ""),
        "metadata_requested_count": metadata.get("requested_count", ""),
        "metadata_actual_count": metadata.get("actual_count", ""),
        "metadata_seed": metadata.get("seed", metadata.get("requested_seed", "")),
        "metadata_chunks": metadata.get("chunks", ""),
        "metadata_duplicates": metadata.get("duplicates", ""),
        "metadata_output_sha256": metadata.get("output_sha256", ""),
    }


def _read_generation_metadata(path: Path) -> dict[str, object]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable_metadata"}
    if not isinstance(payload, dict):
        return {"status": "invalid_metadata"}
    return payload


def _as_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _metadata_has_successful_origin(metadata: dict[str, object], family: str, count: int) -> bool:
    status = str(metadata.get("status", "") or "")
    if status == "generated_chunked":
        pass
    elif status == "reused_existing_output":
        existing_metadata = metadata.get("existing_metadata")
        if not isinstance(existing_metadata, dict):
            return False
        return _metadata_has_successful_origin(existing_metadata, family, count)
    else:
        return False
    if str(metadata.get("family", "") or "") != family:
        return False
    actual_count = _as_int(metadata.get("actual_count", ""))
    return actual_count is not None and actual_count >= count


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw and generated sequence corpus sizes.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "corpus_audit")
    parser.add_argument(
        "--min-generated-per-family",
        type=int,
        default=0,
        help="Fail if any family has fewer generated sequences than this threshold.",
    )
    parser.add_argument(
        "--max-generated-per-family",
        type=int,
        default=0,
        help="Fail if any family has more generated sequences than this threshold. Zero disables the check.",
    )
    args = parser.parse_args()
    if args.min_generated_per_family < 0:
        raise SystemExit("--min-generated-per-family must be non-negative")
    if args.max_generated_per_family < 0:
        raise SystemExit("--max-generated-per-family must be non-negative")
    if (
        args.max_generated_per_family
        and args.min_generated_per_family
        and args.max_generated_per_family < args.min_generated_per_family
    ):
        raise SystemExit("--max-generated-per-family cannot be less than --min-generated-per-family")

    file_rows: list[dict[str, object]] = []
    all_records: list[SequenceRecord] = []
    for family, filename in FAMILY_FILES.items():
        path = args.data_dir / filename
        if path.exists():
            records = read_long_sequences(path, family=family)
            all_records.extend(records)
            file_rows.append(_summarize_records(path, "raw", records, family=family))

    if args.generated_dir.exists():
        for path in sorted(args.generated_dir.glob("*.csv")):
            records = read_long_sequences(path)
            all_records.extend(records)
            file_rows.append(_summarize_records(path, "generated", records))

    total_by_family: Counter[str] = Counter()
    generated_by_family: Counter[str] = Counter()
    steps_by_family: Counter[str] = Counter()
    failures: list[str] = []
    for row in file_rows:
        family = str(row["family"])
        count = int(row["num_sequences"])
        total_by_family[family] += count
        steps_by_family[family] += int(row["num_steps"])
        if row["source"] == "generated":
            generated_by_family[family] += count
            status = str(row.get("metadata_status", "") or "")
            metadata_family = str(row.get("metadata_family", "") or "")
            metadata_actual_count = _as_int(row.get("metadata_actual_count", ""))
            metadata_output_sha256 = str(row.get("metadata_output_sha256", "") or "")
            file_hash = str(row.get("file_sha256", "") or "")
            if not row.get("metadata_present"):
                failures.append(f"Generated file has no metadata sidecar: {row['path']}")
            elif status not in {"generated_chunked", "reused_existing_output"}:
                failures.append(f"Generated file metadata status is not successful for {row['path']}: {status}")
            elif not _metadata_has_successful_origin(
                _read_generation_metadata(Path(str(row["path"]))),
                family,
                count,
            ):
                failures.append(f"Generated file metadata has no successful generation origin: {row['path']}")
            if metadata_family and metadata_family != family:
                failures.append(
                    f"Generated file metadata family mismatch for {row['path']}: "
                    f"{metadata_family} != {family}"
                )
            if metadata_actual_count is not None and metadata_actual_count != count:
                failures.append(
                    f"Generated file metadata actual_count mismatch for {row['path']}: "
                    f"{metadata_actual_count} != {count}"
                )
            if len(metadata_output_sha256) != 64:
                failures.append(f"Generated file metadata is missing output_sha256 for {row['path']}")
            elif metadata_output_sha256 != file_hash:
                failures.append(
                    f"Generated file metadata output_sha256 mismatch for {row['path']}: "
                    f"{metadata_output_sha256} != {file_hash}"
                )

    family_rows: list[dict[str, object]] = []
    for family in sorted(FAMILY_FILES):
        generated_count = generated_by_family[family]
        family_rows.append({
            "family": family,
            "raw_sequences": total_by_family[family] - generated_count,
            "generated_sequences": generated_count,
            "total_sequences": total_by_family[family],
            "total_steps": steps_by_family[family],
            "meets_generated_minimum": generated_count >= args.min_generated_per_family,
            "meets_generated_maximum": (
                True if args.max_generated_per_family == 0 else generated_count <= args.max_generated_per_family
            ),
        })
        if generated_count < args.min_generated_per_family:
            failures.append(
                f"{family} has {generated_count} generated sequences; expected at least {args.min_generated_per_family}"
            )
        if args.max_generated_per_family and generated_count > args.max_generated_per_family:
            failures.append(
                f"{family} has {generated_count} generated sequences; expected at most {args.max_generated_per_family}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.out_dir / "files.csv",
        [
            "path",
            "source",
            "family",
            "num_sequences",
            "num_steps",
            "file_sha256",
            "content_sha256",
            "min_length",
            "max_length",
            "avg_length",
            "metadata_path",
            "metadata_present",
            "metadata_status",
            "metadata_family",
            "metadata_requested_count",
            "metadata_actual_count",
            "metadata_seed",
            "metadata_chunks",
            "metadata_duplicates",
            "metadata_output_sha256",
        ],
        file_rows,
    )
    write_rows(
        args.out_dir / "families.csv",
        [
            "family",
            "raw_sequences",
            "generated_sequences",
            "total_sequences",
            "total_steps",
            "meets_generated_minimum",
            "meets_generated_maximum",
        ],
        family_rows,
    )
    payload = {
        "data_dir": str(args.data_dir),
        "generated_dir": str(args.generated_dir),
        "min_generated_per_family": args.min_generated_per_family,
        "max_generated_per_family": args.max_generated_per_family,
        "corpus_fingerprint": corpus_fingerprint(all_records),
        "families": family_rows,
        "files": file_rows,
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("family,raw_sequences,generated_sequences,total_sequences,meets_generated_minimum,meets_generated_maximum")
    for row in family_rows:
        print(
            f"{row['family']},{row['raw_sequences']},{row['generated_sequences']},"
            f"{row['total_sequences']},{row['meets_generated_minimum']},{row['meets_generated_maximum']}"
        )
    print(f"Wrote {args.out_dir / 'summary.json'}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
