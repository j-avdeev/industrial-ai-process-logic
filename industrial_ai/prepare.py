from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_vocab, load_training_sequences, write_json
from .paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vocab and corpus stats.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "prepared")
    args = parser.parse_args()

    records = load_training_sequences(args.data_dir)
    ensure_dir(args.out_dir)
    payload = build_vocab(records)
    write_json(args.out_dir / "vocab.json", payload)
    print(f"Prepared {payload['num_sequences']} sequences")
    print(f"Vocabulary size: {len(payload['tokens'])}")
    print(f"Wrote {args.out_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()

