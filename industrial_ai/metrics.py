from __future__ import annotations

import argparse
from pathlib import Path

from .data import read_rows, split_steps
from .paths import DEFAULT_SUBMISSIONS_DIR, PROJECT_ROOT


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        curr = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def category(step: str) -> str:
    if "CLEAN" in step or "RCA" in step or step == "HF DIP":
        return "clean"
    if "DEPOSIT" in step or "OXIDATION" in step or "EPITAX" in step:
        return "deposit"
    if "LITHO" in step or "MASK" in step or "PHOTORESIST" in step or "DEVELOP" in step:
        return "litho"
    if "ETCH" in step:
        return "etch"
    if "IMPLANT" in step or "ANNEAL" in step or "DIFFUSION" in step:
        return "implant_anneal"
    if "CMP" in step or "PLANAR" in step:
        return "cmp"
    if "TEST" in step or "SORT" in step or "YIELD" in step:
        return "test"
    if "MEASURE" in step or "INSPECTION" in step or "CHECK" in step:
        return "measure"
    if "SHIP" in step or "LOT" in step:
        return "logistics"
    return "other"


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [(s, y) for s, y in zip(scores, labels) if y == 1]
    negatives = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def eval_nextstep(truth_path: Path, pred_path: Path) -> dict[str, float]:
    truth = {row["EXAMPLE_ID"]: row["NEXT_STEP"] for row in read_rows(truth_path)}
    preds = read_rows(pred_path)
    hits1 = hits3 = hits5 = 0
    rr_total = 0.0
    total = 0
    for row in preds:
        target = truth.get(row["EXAMPLE_ID"])
        if target is None:
            continue
        ranks = [row.get(f"RANK_{idx}", "") for idx in range(1, 6)]
        total += 1
        if ranks[:1] and ranks[0] == target:
            hits1 += 1
        if target in ranks[:3]:
            hits3 += 1
        if target in ranks[:5]:
            hits5 += 1
        if target in ranks:
            rr_total += 1.0 / (ranks.index(target) + 1)
    denom = max(total, 1)
    return {
        "count": float(total),
        "top1": hits1 / denom,
        "top3": hits3 / denom,
        "top5": hits5 / denom,
        "mrr": rr_total / denom,
    }


def eval_completion(truth_path: Path, pred_path: Path) -> dict[str, float]:
    truth = {row["EXAMPLE_ID"]: split_steps(row["REMAINING_SEQUENCE"]) for row in read_rows(truth_path)}
    preds = read_rows(pred_path)
    total = exact = 0
    edit_total = token_acc_total = block_acc_total = 0.0
    for row in preds:
        target = truth.get(row["EXAMPLE_ID"])
        if target is None:
            continue
        pred = split_steps(row.get("PREDICTED_SEQUENCE", ""))
        total += 1
        exact += int(pred == target)
        denom = max(len(target), len(pred), 1)
        edit_total += levenshtein(pred, target) / denom
        max_len = max(len(target), len(pred), 1)
        token_matches = sum(1 for a, b in zip(pred, target) if a == b)
        block_matches = sum(1 for a, b in zip(pred, target) if category(a) == category(b))
        token_acc_total += token_matches / max_len
        block_acc_total += block_matches / max_len
    denom = max(total, 1)
    return {
        "count": float(total),
        "exact_match": exact / denom,
        "normalized_edit_distance": edit_total / denom,
        "token_accuracy": token_acc_total / denom,
        "block_accuracy": block_acc_total / denom,
    }


def eval_anomaly(truth_path: Path, pred_path: Path) -> dict[str, float]:
    truth_rows = {
        row["EXAMPLE_ID"]: (int(row["IS_VALID"]), row.get("RULE", ""))
        for row in read_rows(truth_path)
    }
    tp = tn = fp = fn = total = rule_ok = rule_total = 0
    auc_labels: list[int] = []
    auc_scores: list[float] = []
    for row in read_rows(pred_path):
        if row["EXAMPLE_ID"] not in truth_rows:
            continue
        true_valid, true_rule = truth_rows[row["EXAMPLE_ID"]]
        pred_valid = int(row["IS_VALID"])
        valid_score = float(row.get("SCORE", "0.5") or 0.5)
        total += 1
        if true_valid == 0 and pred_valid == 0:
            tp += 1
        elif true_valid == 1 and pred_valid == 1:
            tn += 1
        elif true_valid == 1 and pred_valid == 0:
            fp += 1
        elif true_valid == 0 and pred_valid == 1:
            fn += 1
        if true_valid == 0 and pred_valid == 0:
            rule_total += 1
            rule_ok += int(row.get("PREDICTED_RULE", "") == true_rule)
        auc_labels.append(1 if true_valid == 0 else 0)
        auc_scores.append(1.0 - valid_score)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "count": float(total),
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "roc_auc": roc_auc(auc_labels, auc_scores),
        "rule_attribution_accuracy": rule_ok / max(rule_total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local dev predictions.")
    parser.add_argument("--dev-dir", type=Path, default=PROJECT_ROOT / "data" / "dev")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    args = parser.parse_args()

    metrics = {
        "nextstep": eval_nextstep(args.dev_dir / "nextstep_truth.csv", args.pred_dir / "nextstep.csv"),
        "completion": eval_completion(args.dev_dir / "completion_truth.csv", args.pred_dir / "completion.csv"),
        "anomaly": eval_anomaly(args.dev_dir / "anomaly_truth.csv", args.pred_dir / "anomaly.csv"),
    }
    for task, values in metrics.items():
        print(f"\n[{task}]")
        for key, value in values.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

