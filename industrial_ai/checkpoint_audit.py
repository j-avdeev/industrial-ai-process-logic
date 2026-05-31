from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT
from .run_profiles import profile_for_count
from .validate_run import _corpus_audit_expectations, _failures_from_checkpoints, _source_bundle_expectation


def _run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    count = max_generated_per_family or min_generated_per_family
    if count <= 0:
        return ""
    return profile_for_count(count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail fast when trained checkpoints do not match the audited generated corpus.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--model-sizes", nargs="*", default=["tiny", "small", "medium"])
    parser.add_argument("--min-generated-per-family", type=int, default=0)
    parser.add_argument("--max-generated-per-family", type=int, default=0)
    parser.add_argument(
        "--require-checkpoint-device",
        default="",
        help="Require every checkpoint train_summary.json to report this actual device.",
    )
    parser.add_argument(
        "--min-train-epochs",
        type=int,
        default=0,
        help="Require every checkpoint train_summary.json to report at least this many epochs.",
    )
    parser.add_argument(
        "--required-batch-size",
        type=int,
        default=0,
        help="Require every checkpoint train_summary.json to report this batch_size. Zero disables the check.",
    )
    parser.add_argument("--out", type=Path, help="Defaults to <artifacts-dir>/checkpoint_audit.json.")
    parser.add_argument(
        "--readiness",
        type=Path,
        help="Leonardo readiness JSON for source-bundle checkpoint provenance. Defaults to <artifacts-dir>/leonardo_readiness.json.",
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

    failures, expected_family_counts, corpus_fingerprint = _corpus_audit_expectations(
        args.artifacts_dir / "corpus_audit" / "summary.json",
        args.min_generated_per_family,
        args.max_generated_per_family,
    )
    readiness_path = args.readiness or (args.artifacts_dir / "leonardo_readiness.json")
    source_bundle_failures, expected_source_bundle_sha256 = _source_bundle_expectation(readiness_path)
    failures.extend(source_bundle_failures)
    failures.extend(_failures_from_checkpoints(
        args.checkpoint_dir,
        args.model_sizes,
        expected_family_counts,
        corpus_fingerprint,
        args.require_checkpoint_device,
        args.min_train_epochs,
        args.required_batch_size,
        expected_source_bundle_sha256,
    ))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "artifacts_dir": str(args.artifacts_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_sizes": args.model_sizes,
        "min_generated_per_family": args.min_generated_per_family,
        "max_generated_per_family": args.max_generated_per_family,
        "run_profile": _run_profile(args.min_generated_per_family, args.max_generated_per_family),
        "required_checkpoint_device": args.require_checkpoint_device,
        "min_train_epochs": args.min_train_epochs,
        "required_batch_size": args.required_batch_size,
        "readiness": str(readiness_path),
        "source_bundle_sha256": expected_source_bundle_sha256,
        "expected_family_counts": expected_family_counts,
        "corpus_fingerprint": corpus_fingerprint,
        "failures": failures,
    }
    out_path = args.out or (args.artifacts_dir / "checkpoint_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    if failures:
        print("Checkpoint audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Checkpoint audit passed")


if __name__ == "__main__":
    main()
