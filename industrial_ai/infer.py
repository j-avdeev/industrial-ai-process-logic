from __future__ import annotations

import argparse
from pathlib import Path

from .anomaly import validate_steps
from .baseline import NGramRanker
from .completion import CompletionEngine, default_checkpoint_path
from .data import load_training_sequences, read_long_sequences, read_rows, split_steps, write_rows
from .paths import DEFAULT_DATA_DIR, DEFAULT_SUBMISSIONS_DIR, PROJECT_ROOT


def _load_corpus(data_dir: Path, generated_dir: Path | None = None):
    records = load_training_sequences(data_dir)
    if generated_dir and generated_dir.exists():
        for path in generated_dir.glob("*.csv"):
            records.extend(read_long_sequences(path))
    return records


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
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--transformer-device", default="cpu")
    args = parser.parse_args()

    records = _load_corpus(args.data_dir, args.generated_dir)
    ranker = NGramRanker(max_order=8).fit(records)
    completion_mode = _resolve_completion_mode(args.completion_mode, args.generated_dir, args.checkpoint)
    completion_engine = CompletionEngine(
        records,
        ranker,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint if args.checkpoint.exists() else None,
        transformer_device=args.transformer_device,
    )
    print(f"Completion mode: {completion_mode}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.valid_input.exists():
        next_rows: list[dict[str, object]] = []
        completion_rows: list[dict[str, object]] = []
        for row in read_rows(args.valid_input):
            example_id = row["EXAMPLE_ID"]
            family = row["FAMILY"].strip().upper()
            partial = split_steps(row["PARTIAL_SEQUENCE"])
            ranks = ranker.rank_next(family, partial, k=5)
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
        print(f"Wrote {args.out_dir / 'anomaly.csv'}")
    else:
        print(f"Skipping anomaly inference; missing {args.anomaly_input}")


if __name__ == "__main__":
    main()
