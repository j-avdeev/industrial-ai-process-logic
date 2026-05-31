from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_vocab, corpus_fingerprint, load_corpus, write_json
from .paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR, PROJECT_ROOT, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vocab and corpus stats.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "prepared")
    args = parser.parse_args()

    records = load_corpus(args.data_dir, args.generated_dir)
    ensure_dir(args.out_dir)
    payload = build_vocab(records)
    payload["data_dir"] = str(args.data_dir)
    payload["generated_dir"] = str(args.generated_dir)
    payload["corpus_fingerprint"] = corpus_fingerprint(records)
    write_json(args.out_dir / "vocab.json", payload)
    print(f"Prepared {payload['num_sequences']} sequences")
    print(f"Vocabulary size: {len(payload['tokens'])}")
    print(f"Wrote {args.out_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()
