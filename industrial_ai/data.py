from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import DEFAULT_DATA_DIR


FAMILY_FILES = {
    "MOSFET": "MOSFET_variants.csv",
    "IGBT": "IGBT_variants.csv",
    "IC": "IC_variants.csv",
}


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    family: str
    steps: tuple[str, ...]


def normalize_family(value: str) -> str:
    family = value.strip().upper()
    if family not in FAMILY_FILES:
        raise ValueError(f"Unknown family {value!r}; expected one of {sorted(FAMILY_FILES)}")
    return family


def split_steps(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def join_steps(steps: Iterable[str]) -> str:
    return "|".join(steps)


def read_long_sequences(path: Path, family: str | None = None) -> list[SequenceRecord]:
    rows: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {name.strip().upper(): name for name in (reader.fieldnames or [])}
        if "STEP" not in fields:
            raise ValueError(f"{path} has no STEP column")
        seq_key = fields.get("SEQUENCE_ID")
        step_key = fields["STEP"]
        for idx, row in enumerate(reader, start=1):
            sid = row.get(seq_key, "").strip() if seq_key else "seq_0001"
            if not sid:
                sid = f"seq_{idx:04d}"
            step = row.get(step_key, "").strip()
            if step:
                rows.setdefault(sid, []).append(step)

    inferred_family = family or infer_family_from_name(path.name)
    return [
        SequenceRecord(sequence_id=sid, family=inferred_family, steps=tuple(steps))
        for sid, steps in rows.items()
    ]


def infer_family_from_name(name: str) -> str:
    upper = name.upper()
    if "MOSFET" in upper:
        return "MOSFET"
    if "IGBT" in upper:
        return "IGBT"
    if "IC" in upper:
        return "IC"
    raise ValueError(f"Cannot infer family from {name!r}")


def load_training_sequences(data_dir: Path | None = None) -> list[SequenceRecord]:
    root = Path(data_dir or DEFAULT_DATA_DIR)
    records: list[SequenceRecord] = []
    for family, filename in FAMILY_FILES.items():
        records.extend(read_long_sequences(root / filename, family=family))
    return records


def load_corpus(
    data_dir: Path | None = None,
    generated_dir: Path | None = None,
) -> list[SequenceRecord]:
    records = load_training_sequences(data_dir)
    if generated_dir:
        generated_root = Path(generated_dir)
        if generated_root.exists():
            for path in sorted(generated_root.glob("*.csv")):
                records.extend(read_long_sequences(path))
    return records


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def corpus_fingerprint(records: Iterable[SequenceRecord]) -> str:
    digest = hashlib.sha256()
    for family, steps in sorted((record.family, tuple(record.steps)) for record in records):
        digest.update(family.encode("utf-8"))
        digest.update(b"\0")
        for step in steps:
            digest.update(step.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def build_vocab(records: Iterable[SequenceRecord]) -> dict[str, object]:
    token_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    lengths: list[int] = []
    for record in records:
        token_counts.update(record.steps)
        family_counts.update([record.family])
        lengths.append(len(record.steps))

    tokens = sorted(token_counts)
    return {
        "tokens": tokens,
        "token_to_id": {token: idx for idx, token in enumerate(tokens)},
        "families": sorted(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "num_sequences": sum(family_counts.values()),
        "num_tokens": sum(token_counts.values()),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }
