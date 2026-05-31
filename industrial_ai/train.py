from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import corpus_fingerprint, load_corpus
from .hashing import file_sha256
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


def _write_loss_curve(log_rows: list[dict[str, float]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [int(row["epoch"]) for row in log_rows]
    losses = [float(row["loss"]) for row in log_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _source_bundle_summary(readiness_path: Path, require_source_bundle_proof: bool) -> dict[str, object]:
    if not require_source_bundle_proof:
        return {
            "required": False,
            "verified": False,
            "bundle_sha256": "",
            "readiness_path": str(readiness_path),
        }
    if not readiness_path.exists():
        raise SystemExit(f"Missing Leonardo readiness source-bundle evidence: {readiness_path}")
    try:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Leonardo readiness source-bundle evidence is not readable: {readiness_path} ({exc})")
    required = readiness.get("require_source_bundle") is True
    source_bundle = readiness.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    bundle_sha256 = str(source_bundle.get("bundle_sha256", "") or "")
    verified = source_bundle.get("verified") is True
    if not required:
        raise SystemExit("Leonardo readiness did not require source-bundle proof")
    if not verified or not bundle_sha256:
        raise SystemExit("Leonardo readiness does not contain a verified source-bundle SHA")
    return {
        "required": True,
        "verified": True,
        "bundle_sha256": bundle_sha256,
        "readiness_path": str(readiness_path),
    }


def _is_complete_training_run(
    run_dir: Path,
    model_size: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    requested_device: str,
    require_device: bool,
    config: dict[str, int],
    records_count: int,
    family_counts: Counter[str],
    fingerprint: str,
    source_bundle_sha256: str,
) -> bool:
    model_path = run_dir / "model.pt"
    summary_path = run_dir / "train_summary.json"
    log_path = run_dir / "train_log.json"
    if not model_path.exists() or not summary_path.exists() or not log_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        log_rows = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("model_size") != model_size:
        return False
    if summary.get("config") != config:
        return False
    if int(summary.get("epochs", -1)) != epochs:
        return False
    if int(summary.get("batch_size", -1)) != batch_size:
        return False
    if float(summary.get("lr", -1.0)) != lr:
        return False
    if int(summary.get("seed", -1)) != seed:
        return False
    if require_device and summary.get("device") != requested_device:
        return False
    if int(summary.get("num_sequences", -1)) != records_count:
        return False
    if summary.get("model_sha256") != file_sha256(model_path):
        return False
    if summary.get("train_log_sha256") != file_sha256(log_path):
        return False
    if summary.get("corpus_fingerprint") != fingerprint:
        return False
    if summary.get("family_counts") != dict(sorted(family_counts.items())):
        return False
    if source_bundle_sha256 and summary.get("source_bundle_sha256") != source_bundle_sha256:
        return False
    if summary.get("final_loss") is None:
        return False
    return isinstance(log_rows, list) and len(log_rows) >= epochs


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
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="Fail instead of falling back when the requested device is unavailable.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Reuse an existing matching checkpoint, train log, and train summary.",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "leonardo_readiness.json",
        help="Leonardo readiness JSON used to bind source-bundle identity into training evidence.",
    )
    parser.add_argument(
        "--require-source-bundle-proof",
        action="store_true",
        help="Fail unless readiness proves a verified source-bundle SHA; matching checkpoints must carry it.",
    )
    args = parser.parse_args()
    requested_device = args.device

    records = load_corpus(args.data_dir, args.generated_dir)
    family_counts = Counter(record.family for record in records)
    fingerprint = corpus_fingerprint(records)
    config = MODEL_CONFIGS[args.model_size]
    source_bundle = _source_bundle_summary(args.readiness, args.require_source_bundle_proof)
    source_bundle_sha256 = str(source_bundle.get("bundle_sha256", "") or "")
    run_dir = args.out_dir / args.model_size
    if args.skip_if_complete and _is_complete_training_run(
        run_dir,
        args.model_size,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        requested_device,
        args.require_device,
        config,
        len(records),
        family_counts,
        fingerprint,
        source_bundle_sha256,
    ):
        print(f"Reusing complete training run at {run_dir}")
        return

    torch, nn = _require_torch()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device_fallback = False
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.require_device:
            raise SystemExit("CUDA was requested with --require-device, but torch.cuda.is_available() is false")
        args.device = "cpu"
        device_fallback = True

    corpus = encode(records)
    pad_id = corpus.token_to_id["<PAD>"]
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

    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_size": args.model_size,
        "config": config,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "requested_device": requested_device,
        "device": args.device,
        "device_fallback": device_fallback,
        "data_dir": str(args.data_dir),
        "generated_dir": str(args.generated_dir),
        "num_sequences": len(records),
        "corpus_fingerprint": fingerprint,
        "family_counts": dict(sorted(family_counts.items())),
        "source_bundle_sha256": source_bundle_sha256,
        "source_bundle_required": source_bundle.get("required") is True,
        "source_bundle_verified": source_bundle.get("verified") is True,
        "source_bundle_readiness_path": source_bundle.get("readiness_path", ""),
        "num_tokens": sum(len(seq) for seq in corpus.sequences),
        "vocab_size": len(corpus.id_to_token),
        "max_len": corpus.max_len,
        "final_loss": log_rows[-1]["loss"] if log_rows else None,
        "final_perplexity": log_rows[-1]["perplexity"] if log_rows else None,
    }
    model_path = run_dir / "model.pt"
    log_path = run_dir / "train_log.json"
    log_path.write_text(json.dumps(log_rows, indent=2), encoding="utf-8")
    summary["train_log_sha256"] = file_sha256(log_path)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "max_len": corpus.max_len,
        "token_to_id": corpus.token_to_id,
        "id_to_token": corpus.id_to_token,
        "model_size": args.model_size,
        "train_summary": summary,
    }, model_path)
    summary["model_sha256"] = file_sha256(model_path)
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if _write_loss_curve(log_rows, run_dir / "loss_curve.png"):
        print(f"Wrote {run_dir / 'loss_curve.png'}")
    else:
        print("Skipped loss curve; matplotlib is not installed")
    print(f"Wrote {run_dir / 'train_summary.json'}")
    print(f"Wrote {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
