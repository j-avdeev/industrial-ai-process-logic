from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .data import SequenceRecord


END_TOKEN = "<END>"


@dataclass
class PrefixMatch:
    record: SequenceRecord
    score: int


class NGramRanker:
    def __init__(self, max_order: int = 6) -> None:
        self.max_order = max_order
        self.counts: dict[tuple[str, int, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
        self.family_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()
        self.vocab: set[str] = set()

    def fit(self, records: Iterable[SequenceRecord]) -> "NGramRanker":
        for record in records:
            seq = list(record.steps) + [END_TOKEN]
            self.vocab.update(record.steps)
            for idx, token in enumerate(seq):
                if token != END_TOKEN:
                    self.global_counts[token] += 1
                    self.family_counts[record.family][token] += 1
                prefix = tuple(seq[:idx])
                for order in range(0, self.max_order + 1):
                    context = prefix[-order:] if order else tuple()
                    self.counts[(record.family, order, context)][token] += 1
        return self

    def rank_next(self, family: str, partial: list[str], k: int = 5) -> list[str]:
        candidates: Counter[str] = Counter()
        for order in range(min(self.max_order, len(partial)), -1, -1):
            context = tuple(partial[-order:]) if order else tuple()
            hits = self.counts.get((family, order, context))
            if hits:
                candidates.update({token: count * (order + 1) for token, count in hits.items()})
                break
        if not candidates:
            candidates.update(self.family_counts.get(family, Counter()))
        if not candidates:
            candidates.update(self.global_counts)

        ranked = [
            token for token, _ in candidates.most_common()
            if token != END_TOKEN and token not in set(partial[-3:])
        ]
        for token, _ in self.global_counts.most_common():
            if token not in ranked:
                ranked.append(token)
            if len(ranked) >= k:
                break
        return ranked[:k]

    def complete_greedy(self, family: str, partial: list[str], max_new_steps: int = 180) -> list[str]:
        steps = list(partial)
        generated: list[str] = []
        for _ in range(max_new_steps):
            next_token = self.rank_next(family, steps, k=1)[0]
            if next_token == END_TOKEN or next_token == "SHIP LOT":
                if next_token == "SHIP LOT" and (not steps or steps[-1] != "SHIP LOT"):
                    generated.append(next_token)
                break
            generated.append(next_token)
            steps.append(next_token)
        return generated


class PrefixIndex:
    def __init__(self, records: Iterable[SequenceRecord]) -> None:
        self.by_family: dict[str, list[SequenceRecord]] = defaultdict(list)
        for record in records:
            self.by_family[record.family].append(record)

    @staticmethod
    def _same_prefix_score(candidate: tuple[str, ...], partial: list[str]) -> int:
        score = 0
        for left, right in zip(candidate, partial):
            if left != right:
                break
            score += 1
        return score

    def best_completion(self, family: str, partial: list[str]) -> list[str] | None:
        matches: list[PrefixMatch] = []
        for record in self.by_family.get(family, []):
            score = self._same_prefix_score(record.steps, partial)
            if score:
                matches.append(PrefixMatch(record=record, score=score))
        if not matches:
            return None
        matches.sort(key=lambda item: (item.score, len(item.record.steps)), reverse=True)
        best = matches[0].record.steps
        cut = min(len(partial), len(best))
        return list(best[cut:])

