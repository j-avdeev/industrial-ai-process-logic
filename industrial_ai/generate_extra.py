from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import read_long_sequences
from .hashing import file_sha256
from .official import load_generator
from .paths import DEFAULT_DATA_DIR


def _existing_sequence_count(path: Path, family: str) -> int:
    if not path.exists():
        return 0
    try:
        return len(read_long_sequences(path, family=family.upper()))
    except (OSError, ValueError):
        return 0


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_metadata(path: Path) -> dict[str, object] | None:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _successful_generation_metadata(
    metadata: dict[str, object] | None,
    family: str,
    count: int,
    exact_count: bool = False,
) -> bool:
    if not metadata:
        return False
    status = str(metadata.get("status", "") or "")
    if status == "generated_chunked":
        pass
    elif status == "reused_existing_output":
        if not _successful_generation_metadata(metadata.get("existing_metadata"), family, count, exact_count):
            return False
    else:
        return False
    if str(metadata.get("family", "") or "") != family.upper():
        return False
    try:
        actual_count = int(metadata.get("actual_count", 0))
    except (TypeError, ValueError):
        return False
    return actual_count == count if exact_count else actual_count >= count


def _metadata_matches_output(metadata: dict[str, object] | None, path: Path) -> bool:
    if not metadata:
        return False
    expected_hash = str(metadata.get("output_sha256", "") or "")
    if len(expected_hash) != 64:
        return False
    return path.exists() and file_sha256(path) == expected_hash


def _sequence_digest(sequence: list[str]) -> str:
    return hashlib.sha1("\0".join(sequence).encode("utf-8")).hexdigest()


def _write_chunked_csv(generator, family: str, count: int, seed: int, chunk_size: int, path: Path) -> dict[str, int]:
    seen: set[str] = set()
    written = 0
    total_steps = 0
    chunk_index = 0
    duplicates = 0
    max_chunks = max(20, ((count + max(chunk_size, 1) - 1) // max(chunk_size, 1)) * 20)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SEQUENCE_ID", "STEP"])
        while written < count and chunk_index < max_chunks:
            remaining = count - written
            request_count = min(chunk_size, remaining)
            chunk = generator.generate_dataset(
                family,
                request_count,
                seed=seed + chunk_index,
                validate=True,
            )
            added = 0
            for sequence in chunk:
                digest = _sequence_digest(sequence)
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                written += 1
                added += 1
                sequence_id = f"seq_{written:06d}"
                for step in sequence:
                    writer.writerow([sequence_id, step])
                total_steps += len(sequence)
                if written >= count:
                    break
            chunk_index += 1
            print(
                f"  chunk={chunk_index} requested={request_count} added={added} "
                f"duplicates={duplicates} total={written}/{count}"
            )

    return {
        "actual_count": written,
        "total_steps": total_steps,
        "chunks": chunk_index,
        "duplicates": duplicates,
        "max_chunks": max_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extra valid sequences with the official grammar.")
    parser.add_argument("--family", choices=["mosfet", "igbt", "ic"], required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Reuse an existing output when it already contains at least --count sequences.",
    )
    parser.add_argument(
        "--exact-count",
        action="store_true",
        help="When reusing output, require exactly --count sequences instead of at least --count.",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    existing_count = _existing_sequence_count(args.output, args.family)
    if args.skip_if_complete and existing_count >= args.count:
        if args.exact_count and existing_count != args.count:
            raise SystemExit(
                f"{args.output} has {existing_count} sequences, but --exact-count requires {args.count}; "
                "remove the stale file or choose the matching --count"
            )
        existing_metadata = _read_metadata(args.output)
        if not _successful_generation_metadata(existing_metadata, args.family, args.count, args.exact_count):
            raise SystemExit(
                f"{args.output} has {existing_count} sequences, but no successful prior generation metadata; "
                "rerun without --skip-if-complete or restore the metadata sidecar"
            )
        if not _metadata_matches_output(existing_metadata, args.output):
            raise SystemExit(
                f"{args.output} has successful metadata, but its output_sha256 does not match the CSV; "
                "rerun without --skip-if-complete or restore the matching metadata sidecar"
            )
        _write_metadata(args.output, {
            "family": args.family.upper(),
            "requested_count": args.count,
            "actual_count": existing_count,
            "exact_count": args.exact_count,
            "requested_seed": args.seed,
            "existing_metadata": existing_metadata,
            "output_sha256": file_sha256(args.output),
            "output": str(args.output),
            "data_dir": str(args.data_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "reused_existing_output",
        })
        print(f"Reusing {args.output}; found {existing_count} {args.family.upper()} sequences")
        return

    generator = load_generator(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_name(f".{args.output.name}.tmp")
    chunk_size = max(1, min(args.chunk_size, args.count))
    stats = _write_chunked_csv(
        generator,
        args.family,
        args.count,
        args.seed,
        chunk_size,
        tmp_path,
    )
    actual_count = stats["actual_count"]
    if actual_count < args.count:
        _write_metadata(args.output, {
            "family": args.family.upper(),
            "requested_count": args.count,
            "actual_count": actual_count,
            "exact_count": args.exact_count,
            "total_steps": stats["total_steps"],
            "seed": args.seed,
            "chunk_size": chunk_size,
            "chunks": stats["chunks"],
            "duplicates": stats["duplicates"],
            "max_chunks": stats["max_chunks"],
            "output": str(args.output),
            "data_dir": str(args.data_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed_unique_sequence_shortfall",
        })
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(f"Only generated {actual_count}/{args.count} unique {args.family.upper()} sequences")
    tmp_path.replace(args.output)
    output_sha256 = file_sha256(args.output)
    _write_metadata(args.output, {
        "family": args.family.upper(),
        "requested_count": args.count,
        "actual_count": actual_count,
        "exact_count": args.exact_count,
        "total_steps": stats["total_steps"],
        "seed": args.seed,
        "chunk_size": chunk_size,
        "chunks": stats["chunks"],
        "duplicates": stats["duplicates"],
        "max_chunks": stats["max_chunks"],
        "output_sha256": output_sha256,
        "output": str(args.output),
        "data_dir": str(args.data_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated_chunked",
    })
    print(f"Wrote {actual_count} {args.family.upper()} sequences to {args.output}")


if __name__ == "__main__":
    main()
