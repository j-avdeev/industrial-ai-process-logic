from __future__ import annotations

import argparse
from pathlib import Path

from .official import load_generator
from .paths import DEFAULT_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extra valid sequences with the official grammar.")
    parser.add_argument("--family", choices=["mosfet", "igbt", "ic"], required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    generator = load_generator(args.data_dir)
    sequences = generator.generate_dataset(args.family, args.count, seed=args.seed, validate=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator.write_csv(args.output, sequences)
    print(f"Wrote {args.count} {args.family.upper()} sequences to {args.output}")


if __name__ == "__main__":
    main()

