from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import NGramRanker
from .completion import CompletionEngine, default_checkpoint_path
from .data import load_training_sequences, read_long_sequences, read_rows, split_steps, write_rows
from .metrics import eval_completion
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


def _load_corpus(data_dir: Path, generated_dir: Path | None = None):
    records = load_training_sequences(data_dir)
    if generated_dir and generated_dir.exists():
        for path in generated_dir.glob("*.csv"):
            records.extend(read_long_sequences(path))
    return records


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
    parser.add_argument("--modes", nargs="+", default=["prefix", "retrieval", "beam", "ensemble"])
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    records = _load_corpus(args.data_dir, args.generated_dir)
    ranker = NGramRanker(max_order=8).fit(records)
    engine = CompletionEngine(
        records,
        ranker,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint if args.checkpoint.exists() else None,
        transformer_device=args.transformer_device,
    )

    input_rows = read_rows(args.dev_dir / "eval_input_valid.csv")
    if args.max_examples > 0:
        input_rows = input_rows[:args.max_examples]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("mode,count,exact_match,normalized_edit_distance,token_accuracy,block_accuracy")
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
        print(
            f"{mode},{metrics['count']:.0f},{metrics['exact_match']:.4f},"
            f"{metrics['normalized_edit_distance']:.4f},{metrics['token_accuracy']:.4f},"
            f"{metrics['block_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

