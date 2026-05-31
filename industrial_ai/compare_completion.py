from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .baseline import NGramRanker
from .completion import CompletionEngine, default_checkpoint_path
from .data import corpus_fingerprint, load_corpus, read_rows, split_steps, write_rows
from .hashing import file_sha256
from .metrics import eval_completion
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


def _fraction(row: dict[str, str]) -> float | None:
    try:
        return float(row.get("COMPLETION_FRACTION", "") or 0.0)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completion strategies on a local dev set.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--dev-dir", type=Path, default=PROJECT_ROOT / "data" / "dev")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "completion_compare")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--transformer-device", default="cpu")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-transformer-available", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["prefix", "retrieval", "beam", "ensemble"])
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    records = load_corpus(args.data_dir, args.generated_dir)
    corpus_family_counts = Counter(record.family for record in records)
    ranker = NGramRanker(max_order=8).fit(records)
    if args.require_checkpoint and not args.checkpoint.exists():
        raise SystemExit(f"Required checkpoint does not exist: {args.checkpoint}")
    engine = CompletionEngine(
        records,
        ranker,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint if args.checkpoint.exists() else None,
        transformer_device=args.transformer_device,
    )
    transformer_available = bool(engine.scorer and engine.scorer.available)
    if args.require_transformer_available and not transformer_available:
        raise SystemExit(f"Required transformer scorer is not available for checkpoint: {args.checkpoint}")

    input_rows = read_rows(args.dev_dir / "eval_input_valid.csv")
    if args.max_examples > 0:
        input_rows = input_rows[:args.max_examples]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("mode,count,exact_match,normalized_edit_distance,token_accuracy,block_accuracy")
    metric_rows: list[dict[str, object]] = []
    for mode in args.modes:
        pred_path = args.out_dir / f"completion_{mode}.csv"
        rows = []
        for row in input_rows:
            suffix = engine.complete(
                row["FAMILY"].strip().upper(),
                split_steps(row["PARTIAL_SEQUENCE"]),
                completion_fraction=_fraction(row),
                mode=mode,
            )
            rows.append({"EXAMPLE_ID": row["EXAMPLE_ID"], "PREDICTED_SEQUENCE": "|".join(suffix)})
        write_rows(pred_path, ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], rows)
        metrics = eval_completion(args.dev_dir / "completion_truth.csv", pred_path)
        metric_rows.append({
            "mode": mode,
            "prediction_path": str(pred_path),
            **metrics,
        })
        print(
            f"{mode},{metrics['count']:.0f},{metrics['exact_match']:.4f},"
            f"{metrics['normalized_edit_distance']:.4f},{metrics['token_accuracy']:.4f},"
            f"{metrics['block_accuracy']:.4f}"
        )

    write_rows(
        args.out_dir / "metrics.csv",
        ["mode", "count", "exact_match", "normalized_edit_distance", "token_accuracy", "block_accuracy", "prediction_path"],
        metric_rows,
    )
    summary = {
        "data_dir": str(args.data_dir),
        "generated_dir": str(args.generated_dir),
        "dev_dir": str(args.dev_dir),
        "num_corpus_sequences": len(records),
        "corpus_fingerprint": corpus_fingerprint(records),
        "corpus_family_counts": dict(sorted(corpus_family_counts.items())),
        "requested_checkpoint": str(args.checkpoint),
        "checkpoint_used": str(args.checkpoint) if args.checkpoint.exists() else "",
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint.exists() else "",
        "transformer_device": args.transformer_device,
        "transformer_available": transformer_available,
        "require_checkpoint": args.require_checkpoint,
        "require_transformer_available": args.require_transformer_available,
        "modes": metric_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'metrics.csv'}")
    print(f"Wrote {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
