from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .baseline import END_TOKEN, NGramRanker
from .data import SequenceRecord
from .official import load_generator
from .paths import PROJECT_ROOT


@dataclass(frozen=True)
class CompletionCandidate:
    suffix: tuple[str, ...]
    score: float
    method: str
    source_id: str
    context_len: int
    valid: bool
    rule: str


class TransformerSuffixScorer:
    """Optional teacher-forced scorer for candidate suffixes.

    The scorer is intentionally lazy: if PyTorch or a checkpoint is unavailable,
    completion still works with retrieval, n-gram beam search, and validation.
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.available = False
        self.device = device
        self.model = None
        self.torch = None
        self.token_to_id: dict[str, int] = {}
        self.max_len = 0

        if not checkpoint_path.exists():
            return
        try:
            import torch

            from .train import build_model

            from torch import nn
            payload = torch.load(checkpoint_path, map_location=device)
            self.torch = torch
            self.token_to_id = payload["token_to_id"]
            self.max_len = int(payload["max_len"])
            self.model = build_model(
                torch,
                nn,
                len(payload["id_to_token"]),
                self.max_len,
                payload["config"],
            )
            self.model.load_state_dict(payload["model_state"])
            self.model.to(device)
            self.model.eval()
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"[WARN] Transformer scorer disabled: {exc}")

    def score(self, family: str, partial: list[str], suffix: tuple[str, ...]) -> float:
        if not self.available or self.model is None or self.torch is None:
            return 0.0

        family_token = f"<FAMILY_{family}>"
        if family_token not in self.token_to_id:
            return 0.0
        tokens = ["<BOS>", family_token] + partial + list(suffix) + ["<EOS>"]
        if any(token not in self.token_to_id for token in tokens):
            return -25.0
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]

        ids = [self.token_to_id[token] for token in tokens]
        if len(ids) < 3:
            return 0.0

        torch = self.torch
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        prefix_len = max(0, len(partial) + 1)
        try:
            with torch.no_grad():
                logits = self.model(x)[0]
                log_probs = torch.log_softmax(logits, dim=-1)
                suffix_positions = list(range(min(prefix_len, len(y)), len(y)))
                if not suffix_positions:
                    return 0.0
                total = 0.0
                for pos in suffix_positions:
                    total += float(log_probs[pos, y[pos]].item())
                return total / max(len(suffix_positions), 1)
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"[WARN] Transformer scoring failed once: {exc}")
            self.available = False
            return 0.0


class CompletionEngine:
    def __init__(
        self,
        records: Iterable[SequenceRecord],
        ranker: NGramRanker,
        data_dir: Path,
        checkpoint_path: Path | None = None,
        transformer_device: str = "cpu",
    ) -> None:
        self.by_family: dict[str, list[SequenceRecord]] = defaultdict(list)
        self.lengths: dict[str, list[int]] = defaultdict(list)
        self.ranker = ranker
        self.validator = load_generator(data_dir).validate_sequence
        for record in records:
            self.by_family[record.family].append(record)
            self.lengths[record.family].append(len(record.steps))
        self.scorer = TransformerSuffixScorer(checkpoint_path, transformer_device) if checkpoint_path else None

    def complete(
        self,
        family: str,
        partial: list[str],
        completion_fraction: float | None = None,
        mode: str = "ensemble",
        max_new_steps: int = 180,
        top_records: int = 64,
        beam_width: int = 8,
    ) -> list[str]:
        if mode == "prefix":
            return self._best_exact_prefix(family, partial) or self.ranker.complete_greedy(family, partial, max_new_steps)

        candidates = self.retrieve_candidates(family, partial, completion_fraction, top_records=top_records)
        greedy_suffix = tuple(self.ranker.complete_greedy(family, partial, max_new_steps))
        if greedy_suffix:
            valid, rule = self._validate(partial, greedy_suffix)
            candidates.append(CompletionCandidate(
                suffix=greedy_suffix,
                score=220.0,
                method="ngram-greedy",
                source_id="ngram-greedy",
                context_len=0,
                valid=valid,
                rule=rule,
            ))
        if mode in {"ensemble", "beam"}:
            candidates.extend(self.beam_candidates(family, partial, completion_fraction, max_new_steps, beam_width))
        if not candidates:
            return self.ranker.complete_greedy(family, partial, max_new_steps)

        reranked = self.rerank(family, partial, candidates, completion_fraction)
        return list(reranked[0].suffix)

    def retrieve_candidates(
        self,
        family: str,
        partial: list[str],
        completion_fraction: float | None,
        top_records: int = 64,
    ) -> list[CompletionCandidate]:
        expected_total = self._expected_total_length(family, partial, completion_fraction)
        rough: list[tuple[float, int, int, SequenceRecord, int]] = []
        for idx, record in enumerate(self.by_family.get(family, [])):
            exact_prefix = self._exact_prefix_len(record.steps, partial)
            context_len, context_end = self._best_context(record.steps, partial, expected_total)
            if exact_prefix == 0 and context_len == 0:
                continue

            suffix_start = len(partial) if exact_prefix == len(partial) else context_end
            remaining = max(0, len(record.steps) - suffix_start)
            length_penalty = abs(len(record.steps) - expected_total) * 0.75
            remaining_penalty = abs(remaining - max(0, expected_total - len(partial))) * 0.35
            score = exact_prefix * 8.0 + context_len * 24.0 - length_penalty - remaining_penalty
            rough.append((score, idx, suffix_start, record, context_len))

        rough.sort(key=lambda item: item[0], reverse=True)
        candidates: list[CompletionCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for score, _, suffix_start, record, context_len in rough[:top_records]:
            suffix = tuple(record.steps[suffix_start:])
            if suffix in seen:
                continue
            seen.add(suffix)
            valid, rule = self._validate(partial, suffix)
            candidates.append(CompletionCandidate(
                suffix=suffix,
                score=score,
                method="retrieval",
                source_id=record.sequence_id,
                context_len=context_len,
                valid=valid,
                rule=rule,
            ))
        return candidates

    def beam_candidates(
        self,
        family: str,
        partial: list[str],
        completion_fraction: float | None,
        max_new_steps: int,
        beam_width: int,
    ) -> list[CompletionCandidate]:
        expected_total = self._expected_total_length(family, partial, completion_fraction)
        target_remaining = max(1, min(max_new_steps, expected_total - len(partial)))
        beams: list[tuple[float, tuple[str, ...], tuple[str, ...]]] = [(0.0, tuple(), tuple(partial))]
        finished: list[tuple[float, tuple[str, ...]]] = []

        for step_idx in range(min(max_new_steps, max(target_remaining + 30, 25))):
            next_beams: list[tuple[float, tuple[str, ...], tuple[str, ...]]] = []
            for score, suffix, context in beams:
                ranked = self._rank_with_scores(family, list(context), k=beam_width)
                for token, token_score in ranked:
                    if token == END_TOKEN:
                        finished.append((score + token_score - abs(len(suffix) - target_remaining) * 0.6, suffix))
                        continue
                    if len(suffix) < max(1, target_remaining // 3) and token == "SHIP LOT":
                        continue
                    if suffix[-3:].count(token) >= 2:
                        continue
                    new_suffix = suffix + (token,)
                    new_context = context + (token,)
                    length_pressure = -abs(len(new_suffix) - target_remaining) * 0.015
                    ship_bonus = 12.0 if token == "SHIP LOT" else 0.0
                    next_beams.append((score + token_score + length_pressure + ship_bonus, new_suffix, new_context))
                    if token == "SHIP LOT":
                        finished.append((score + token_score + ship_bonus, new_suffix))
            if not next_beams:
                break
            beams = heapq.nlargest(beam_width, next_beams, key=lambda item: item[0])
            if finished and step_idx >= max(4, int(target_remaining * 0.7)):
                break

        for score, suffix, _ in beams:
            if suffix:
                finished.append((score - abs(len(suffix) - target_remaining) * 0.5, suffix))

        candidates: list[CompletionCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for score, suffix in heapq.nlargest(beam_width * 2, finished, key=lambda item: item[0]):
            if suffix in seen:
                continue
            seen.add(suffix)
            valid, rule = self._validate(partial, suffix)
            candidates.append(CompletionCandidate(
                suffix=suffix,
                score=score,
                method="beam",
                source_id="ngram-beam",
                context_len=0,
                valid=valid,
                rule=rule,
            ))
        return candidates

    def rerank(
        self,
        family: str,
        partial: list[str],
        candidates: list[CompletionCandidate],
        completion_fraction: float | None,
    ) -> list[CompletionCandidate]:
        expected_total = self._expected_total_length(family, partial, completion_fraction)
        base_scores: list[tuple[CompletionCandidate, float]] = []
        for candidate in candidates:
            total_len = len(partial) + len(candidate.suffix)
            length_penalty = abs(total_len - expected_total) * 1.2
            valid_bonus = 250.0 if candidate.valid else -400.0
            ship_bonus = 60.0 if candidate.suffix and candidate.suffix[-1] == "SHIP LOT" else -35.0
            nonempty_bonus = 25.0 if candidate.suffix else -200.0
            score = candidate.score + valid_bonus + ship_bonus + nonempty_bonus - length_penalty
            base_scores.append((candidate, score))

        transformer_targets: set[tuple[str, ...]] = set()
        if self.scorer is not None and self.scorer.available:
            top_for_transformer = sorted(base_scores, key=lambda item: item[1], reverse=True)[:16]
            transformer_targets = {candidate.suffix for candidate, _ in top_for_transformer}

        rescored: list[CompletionCandidate] = []
        for candidate, score in base_scores:
            if candidate.suffix in transformer_targets and self.scorer is not None and self.scorer.available:
                score += self.scorer.score(family, partial, candidate.suffix) * 18.0
            rescored.append(CompletionCandidate(
                suffix=candidate.suffix,
                score=score,
                method=candidate.method,
                source_id=candidate.source_id,
                context_len=candidate.context_len,
                valid=candidate.valid,
                rule=candidate.rule,
            ))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    def _best_exact_prefix(self, family: str, partial: list[str]) -> list[str] | None:
        best: SequenceRecord | None = None
        for record in self.by_family.get(family, []):
            if list(record.steps[:len(partial)]) == partial:
                if best is None or len(record.steps) > len(best.steps):
                    best = record
        if best is None:
            return None
        return list(best.steps[len(partial):])

    @staticmethod
    def _exact_prefix_len(candidate: tuple[str, ...], partial: list[str]) -> int:
        score = 0
        for left, right in zip(candidate, partial):
            if left != right:
                break
            score += 1
        return score

    @staticmethod
    def _best_context(candidate: tuple[str, ...], partial: list[str], expected_total: int) -> tuple[int, int]:
        if not partial:
            return 0, 0
        best_len = 0
        best_end = 0
        max_context = min(18, len(partial), len(candidate))
        expected_pos = min(len(candidate), max(1, len(partial)))
        if expected_total > 0:
            expected_pos = int(len(candidate) * (len(partial) / expected_total))
        for size in range(max_context, 0, -1):
            needle = tuple(partial[-size:])
            for start in range(0, len(candidate) - size + 1):
                if candidate[start:start + size] != needle:
                    continue
                end = start + size
                position_penalty = abs(end - expected_pos)
                if size > best_len or (size == best_len and position_penalty < abs(best_end - expected_pos)):
                    best_len = size
                    best_end = end
            if best_len:
                break
        return best_len, best_end

    def _expected_total_length(self, family: str, partial: list[str], completion_fraction: float | None) -> int:
        if completion_fraction and completion_fraction > 0:
            return max(len(partial) + 1, int(round(len(partial) / completion_fraction)))
        lengths = self.lengths.get(family)
        if not lengths:
            return len(partial) + 60
        return max(len(partial) + 1, int(round(mean(lengths))))

    def _rank_with_scores(self, family: str, partial: list[str], k: int) -> list[tuple[str, float]]:
        scores: Counter[str] = Counter()
        for order in range(min(self.ranker.max_order, len(partial)), -1, -1):
            context = tuple(partial[-order:]) if order else tuple()
            hits = self.ranker.counts.get((family, order, context))
            if hits:
                total = sum(hits.values())
                for token, count in hits.most_common(k * 3):
                    scores[token] = math.log((count + 1) / (total + len(hits))) + order * 0.18
                break
        if not scores:
            hits = self.ranker.family_counts.get(family) or self.ranker.global_counts
            total = sum(hits.values())
            for token, count in hits.most_common(k * 3):
                scores[token] = math.log((count + 1) / (total + len(hits)))
        return scores.most_common(k)

    def _validate(self, partial: list[str], suffix: tuple[str, ...]) -> tuple[bool, str]:
        violations = self.validator(list(partial) + list(suffix))
        if not violations:
            return True, ""
        return False, violations[0].rule


def default_checkpoint_path() -> Path:
    for size in ("medium", "small", "tiny"):
        path = PROJECT_ROOT / "checkpoints" / size / "model.pt"
        if path.exists():
            return path
    return PROJECT_ROOT / "checkpoints" / "tiny" / "model.pt"
