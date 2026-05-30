from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .data import load_training_sequences, read_long_sequences
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
MODEL_CONFIGS = {
    "tiny": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 192},
    "small": {"d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 384},
    "medium": {"d_model": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 768},
}


@dataclass
class EncodedCorpus:
    sequences: list[list[int]]
    token_to_id: dict[str, int]
    id_to_token: list[str]
    max_len: int


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install requirements.txt or use the Leonardo module stack."
        ) from exc
    return torch, nn


def load_corpus(data_dir: Path, generated_dir: Path | None = None):
    records = load_training_sequences(data_dir)
    if generated_dir and generated_dir.exists():
        for path in generated_dir.glob("*.csv"):
            records.extend(read_long_sequences(path))
    return records


def encode(records) -> EncodedCorpus:
    tokens = list(SPECIAL_TOKENS)
    for family in ("IC", "IGBT", "MOSFET"):
        tokens.append(f"<FAMILY_{family}>")
    seen = set(tokens)
    for record in records:
        for step in record.steps:
            if step not in seen:
                seen.add(step)
                tokens.append(step)
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    sequences = []
    for record in records:
        ids = [token_to_id["<BOS>"], token_to_id[f"<FAMILY_{record.family}>"]]
        ids.extend(token_to_id[step] for step in record.steps)
        ids.append(token_to_id["<EOS>"])
        sequences.append(ids)
    return EncodedCorpus(sequences=sequences, token_to_id=token_to_id, id_to_token=tokens, max_len=max(map(len, sequences)))


class StepTransformer:  # placeholder for type checkers; actual class is built after torch import
    pass


def build_model(torch, nn, vocab_size: int, max_len: int, config: dict[str, int]):
    class _StepTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, config["d_model"])
            self.pos_embed = nn.Embedding(max_len, config["d_model"])
            layer = nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                batch_first=True,
                dropout=0.1,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config["num_layers"])
            self.head = nn.Linear(config["d_model"], vocab_size)

        def forward(self, x):
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            hidden = self.token_embed(x) + self.pos_embed(positions)
            mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
            return self.head(self.encoder(hidden, mask=mask))

    return _StepTransformer()


def make_batches(torch, sequences: list[list[int]], batch_size: int, pad_id: int, rng: random.Random):
    order = list(range(len(sequences)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [sequences[idx] for idx in order[start:start + batch_size]]
        max_len = max(len(seq) for seq in batch)
        x = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
        y = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
        for row, seq in enumerate(batch):
            x[row, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            y[row, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        yield x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a step-token transformer.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="tiny")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch, nn = _require_torch()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    records = load_corpus(args.data_dir, args.generated_dir)
    corpus = encode(records)
    pad_id = corpus.token_to_id["<PAD>"]
    config = MODEL_CONFIGS[args.model_size]
    model = build_model(torch, nn, len(corpus.id_to_token), corpus.max_len, config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for x, y in make_batches(torch, corpus.sequences, args.batch_size, pad_id, rng):
            x = x.to(args.device)
            y = y.to(args.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            count = int((y != pad_id).sum().item())
            total_loss += float(loss.item()) * count
            total_tokens += count
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        log_rows.append({"epoch": epoch, "loss": avg_loss, "perplexity": ppl})
        print(f"epoch={epoch} loss={avg_loss:.4f} ppl={ppl:.2f}")

    run_dir = args.out_dir / args.model_size
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "max_len": corpus.max_len,
        "token_to_id": corpus.token_to_id,
        "id_to_token": corpus.id_to_token,
        "model_size": args.model_size,
    }, run_dir / "model.pt")
    (run_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2), encoding="utf-8")
    print(f"Wrote {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

