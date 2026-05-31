from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .anomaly import validate_steps
from .baseline import NGramRanker
from .completion import CompletionEngine, default_checkpoint_path
from .data import corpus_fingerprint, load_corpus, read_rows, split_steps, write_rows
from .hashing import file_sha256
from .paths import DEFAULT_DATA_DIR, DEFAULT_SUBMISSIONS_DIR, PROJECT_ROOT


def _has_generated_data(path: Path) -> bool:
    return path.exists() and any(path.glob("*.csv"))


def _resolve_completion_mode(mode: str, generated_dir: Path, checkpoint_path: Path) -> str:
    if mode != "auto":
        return mode
    if _has_generated_data(generated_dir) or checkpoint_path.exists():
        return "ensemble"
    return "prefix"


def _completion_fraction(row: dict[str, str]) -> float | None:
    value = row.get("COMPLETION_FRACTION", "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _read_selected_checkpoint(path: Path) -> tuple[Path | None, str, list[str]]:
    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, "", [f"Selected reranker metrics are not readable: {path} ({exc})"]

    best_reranker = str(payload.get("best_reranker", "") or "")
    best_checkpoint_text = str(payload.get("best_checkpoint", "") or "").strip()
    if not best_checkpoint_text:
        failures.append("Reranker metrics did not select a checkpoint")
        return None, best_reranker, failures
    best_checkpoint = _resolve_path(Path(best_checkpoint_text))
    best_row = None
    for row in payload.get("runs", []):
        if isinstance(row, dict) and str(row.get("reranker", "") or "") == best_reranker:
            best_row = row
            break
    if best_row is None:
        failures.append(f"Selected reranker row is missing from metrics: {best_reranker}")
    else:
        if best_row.get("available") is not True:
            failures.append(f"Selected reranker was not available: {best_reranker}")
        if "selection_eligible" in best_row and best_row.get("selection_eligible") is not True:
            failures.append(f"Selected reranker was not selection eligible: {best_reranker}")
        row_checkpoint = str(best_row.get("checkpoint", "") or "").strip()
        if not row_checkpoint:
            failures.append(f"Selected reranker row has no checkpoint: {best_reranker}")
        elif _resolve_path(Path(row_checkpoint)) != best_checkpoint:
            failures.append(
                f"Selected reranker checkpoint {row_checkpoint!r} does not match best_checkpoint {best_checkpoint_text!r}"
            )
        expected_sha = str(best_row.get("checkpoint_sha256", "") or "").strip()
        if not expected_sha:
            failures.append(f"Selected reranker row has no checkpoint_sha256: {best_reranker}")
        elif not best_checkpoint.exists():
            failures.append(f"Selected checkpoint does not exist: {best_checkpoint}")
        else:
            actual_sha = file_sha256(best_checkpoint)
            if actual_sha != expected_sha:
                failures.append(
                    f"Selected checkpoint hash mismatch for {best_checkpoint}: "
                    f"metrics checkpoint_sha256={expected_sha} actual={actual_sha}"
                )
    return best_checkpoint, best_reranker, failures


def _resolve_inference_checkpoint(
    requested_checkpoint: Path | None,
    require_selected_checkpoint: bool,
    selected_checkpoint: Path | None,
) -> tuple[Path, list[str]]:
    if not require_selected_checkpoint:
        return requested_checkpoint or default_checkpoint_path(), []
    if selected_checkpoint is None:
        return requested_checkpoint or default_checkpoint_path(), ["Reranker metrics did not select a checkpoint"]
    if requested_checkpoint is None:
        return selected_checkpoint, []
    requested_resolved = _resolve_path(requested_checkpoint)
    if requested_resolved != selected_checkpoint:
        return requested_checkpoint, [
            f"Requested checkpoint {requested_resolved} does not match selected checkpoint {selected_checkpoint}"
        ]
    return requested_checkpoint, []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Track 1 submission CSVs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--valid-input", type=Path, default=PROJECT_ROOT / "data" / "dev" / "eval_input_valid.csv")
    parser.add_argument("--anomaly-input", type=Path, default=PROJECT_ROOT / "data" / "dev" / "eval_input_anomaly.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument("--max-new-steps", type=int, default=180)
    parser.add_argument(
        "--completion-mode",
        choices=["auto", "prefix", "retrieval", "beam", "ensemble"],
        default="auto",
        help="Completion strategy. auto uses ensemble when generated data/checkpoints exist, otherwise prefix baseline.",
    )
    parser.add_argument("--completion-top-records", type=int, default=96)
    parser.add_argument("--completion-beam-width", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reranker-metrics", type=Path, default=PROJECT_ROOT / "artifacts" / "reranker_compare" / "metrics.json")
    parser.add_argument("--transformer-device", default="cpu")
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail if the requested/default checkpoint path does not exist.",
    )
    parser.add_argument(
        "--require-transformer-available",
        action="store_true",
        help="Fail if the checkpoint transformer scorer cannot be loaded.",
    )
    parser.add_argument(
        "--require-selected-checkpoint",
        action="store_true",
        help="Fail if --checkpoint does not match the selected checkpoint in reranker metrics.",
    )
    args = parser.parse_args()

    selected_checkpoint: Path | None = None
    selected_reranker = ""
    if args.require_selected_checkpoint:
        selected_checkpoint, selected_reranker, selection_failures = _read_selected_checkpoint(args.reranker_metrics)
        if selection_failures:
            raise SystemExit("; ".join(selection_failures))
    args.checkpoint, checkpoint_failures = _resolve_inference_checkpoint(
        args.checkpoint,
        args.require_selected_checkpoint,
        selected_checkpoint,
    )
    if checkpoint_failures:
        raise SystemExit("; ".join(checkpoint_failures))

    records = load_corpus(args.data_dir, args.generated_dir)
    ranker = NGramRanker(max_order=8).fit(records)
    completion_mode = _resolve_completion_mode(args.completion_mode, args.generated_dir, args.checkpoint)
    if args.require_checkpoint and not args.checkpoint.exists():
        raise SystemExit(f"Required checkpoint does not exist: {args.checkpoint}")
    completion_engine = CompletionEngine(
        records,
        ranker,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint if args.checkpoint.exists() else None,
        transformer_device=args.transformer_device,
    )
    transformer_available = bool(completion_engine.scorer and completion_engine.scorer.available)
    if args.require_transformer_available and not transformer_available:
        raise SystemExit(f"Required transformer scorer is not available for checkpoint: {args.checkpoint}")
    print(f"Completion mode: {completion_mode}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "generated_dir": str(args.generated_dir),
        "valid_input": str(args.valid_input),
        "anomaly_input": str(args.anomaly_input),
        "out_dir": str(args.out_dir),
        "completion_mode": completion_mode,
        "requested_checkpoint": str(args.checkpoint),
        "checkpoint_used": str(args.checkpoint) if args.checkpoint.exists() else "",
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint.exists() else "",
        "reranker_metrics": str(args.reranker_metrics),
        "selected_reranker": selected_reranker,
        "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint else "",
        "transformer_device": args.transformer_device,
        "transformer_available": transformer_available,
        "require_checkpoint": args.require_checkpoint,
        "require_transformer_available": args.require_transformer_available,
        "require_selected_checkpoint": args.require_selected_checkpoint,
        "num_corpus_sequences": len(records),
        "corpus_fingerprint": corpus_fingerprint(records),
        "nextstep_rows": 0,
        "completion_rows": 0,
        "anomaly_rows": 0,
    }

    if args.valid_input.exists():
        next_rows: list[dict[str, object]] = []
        completion_rows: list[dict[str, object]] = []
        for row in read_rows(args.valid_input):
            example_id = row["EXAMPLE_ID"]
            family = row["FAMILY"].strip().upper()
            partial = split_steps(row["PARTIAL_SEQUENCE"])
            ranks = ranker.rank_next(family, partial, k=max(12, 5))
            ranks = completion_engine.rerank_next(family, partial, ranks, k=5)
            ranks = ranks + [""] * (5 - len(ranks))
            next_rows.append({
                "EXAMPLE_ID": example_id,
                "RANK_1": ranks[0],
                "RANK_2": ranks[1],
                "RANK_3": ranks[2],
                "RANK_4": ranks[3],
                "RANK_5": ranks[4],
            })
            suffix = completion_engine.complete(
                family,
                partial,
                completion_fraction=_completion_fraction(row),
                mode=completion_mode,
                max_new_steps=args.max_new_steps,
                top_records=args.completion_top_records,
                beam_width=args.completion_beam_width,
            )
            completion_rows.append({"EXAMPLE_ID": example_id, "PREDICTED_SEQUENCE": "|".join(suffix)})

        write_rows(args.out_dir / "nextstep.csv", ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], next_rows)
        write_rows(args.out_dir / "completion.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], completion_rows)
        summary["nextstep_rows"] = len(next_rows)
        summary["completion_rows"] = len(completion_rows)
        print(f"Wrote {args.out_dir / 'nextstep.csv'}")
        print(f"Wrote {args.out_dir / 'completion.csv'}")
    else:
        print(f"Skipping valid-task inference; missing {args.valid_input}")

    if args.anomaly_input.exists():
        anomaly_rows: list[dict[str, object]] = []
        for row in read_rows(args.anomaly_input):
            is_valid, score, rule = validate_steps(split_steps(row["SEQUENCE"]), args.data_dir)
            anomaly_rows.append({
                "EXAMPLE_ID": row["EXAMPLE_ID"],
                "IS_VALID": 1 if is_valid else 0,
                "SCORE": f"{score:.4f}",
                "PREDICTED_RULE": rule,
            })
        write_rows(args.out_dir / "anomaly.csv", ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"], anomaly_rows)
        summary["anomaly_rows"] = len(anomaly_rows)
        print(f"Wrote {args.out_dir / 'anomaly.csv'}")
    else:
        print(f"Skipping anomaly inference; missing {args.anomaly_input}")

    (args.out_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'inference_summary.json'}")


if __name__ == "__main__":
    main()
