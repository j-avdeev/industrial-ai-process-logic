from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .baseline import NGramRanker
from .completion import CompletionEngine
from .data import corpus_fingerprint, load_corpus, read_rows, split_steps, write_rows
from .hashing import file_sha256
from .metrics import category, eval_completion, eval_nextstep, levenshtein
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


def _fraction(row: dict[str, str]) -> float | None:
    try:
        return float(row.get("COMPLETION_FRACTION", "") or 0.0)
    except ValueError:
        return None


def _default_checkpoints() -> list[Path]:
    return [
        PROJECT_ROOT / "checkpoints" / "tiny" / "model.pt",
        PROJECT_ROOT / "checkpoints" / "small" / "model.pt",
        PROJECT_ROOT / "checkpoints" / "medium" / "model.pt",
    ]


def _checkpoint_label(path: Path | None) -> str:
    if path is None:
        return "baseline"
    if path.parent.name:
        return path.parent.name
    return path.stem


def _selection_score(nextstep: dict[str, float], completion: dict[str, float]) -> float:
    return (
        nextstep["mrr"]
        + nextstep["top5"] * 0.25
        + completion["token_accuracy"]
        + completion["block_accuracy"] * 0.5
        - completion["normalized_edit_distance"]
        + completion["exact_match"] * 2.0
    )


def _write_report(
    out_dir: Path,
    metric_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    plots: list[str],
) -> None:
    best = metric_rows[0] if metric_rows else None
    lines = [
        "# Reranker Comparison",
        "",
        f"Best reranker: `{best['reranker']}`" if best else "Best reranker: n/a",
        f"Best checkpoint: `{best['checkpoint']}`" if best and best.get("checkpoint") else "Best checkpoint: baseline/no checkpoint",
        f"Selection scope: `{best.get('selection_scope', 'all')}`" if best else "Selection scope: n/a",
        "",
        "## Overall Metrics",
        "",
        "| Reranker | Available | Eligible | Next MRR | Next Top-1 | Completion Exact | Completion Token Acc | Selection Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['reranker']} | {row['available']} | {row['selection_eligible']} | "
            f"{float(row['nextstep_mrr']):.4f} | "
            f"{float(row['nextstep_top1']):.4f} | {float(row['completion_exact_match']):.4f} | "
            f"{float(row['completion_token_accuracy']):.4f} | {float(row['selection_score']):.4f} |"
        )
    lines.extend([
        "",
        "## Per-Family Metrics",
        "",
        "| Reranker | Family | Next MRR | Completion Exact | Completion Token Acc | Selection Score |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(family_rows, key=lambda item: (str(item["family"]), str(item["reranker"]))):
        lines.append(
            f"| {row['reranker']} | {row['family']} | {float(row['nextstep_mrr']):.4f} | "
            f"{float(row['completion_exact_match']):.4f} | "
            f"{float(row['completion_token_accuracy']):.4f} | {float(row['selection_score']):.4f} |"
        )
    if plots:
        lines.extend(["", "## Plots", ""])
        for plot in plots:
            lines.append(f"- [{plot}]({plot})")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out_dir: Path, metric_rows: list[dict[str, object]], family_rows: list[dict[str, object]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plots: list[str] = []
    labels = [str(row["reranker"]) for row in metric_rows]
    if labels:
        score_path = out_dir / "selection_scores.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, [float(row["selection_score"]) for row in metric_rows], color="#4C78A8")
        ax.set_ylabel("selection score")
        ax.set_title("Reranker selection score")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(score_path, dpi=160)
        plt.close(fig)
        plots.append(score_path.name)

        metrics_path = out_dir / "task_metrics.png"
        width = 0.25
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([idx - width for idx in x], [float(row["nextstep_mrr"]) for row in metric_rows], width, label="next MRR")
        ax.bar(x, [float(row["completion_token_accuracy"]) for row in metric_rows], width, label="completion token acc")
        ax.bar([idx + width for idx in x], [float(row["completion_block_accuracy"]) for row in metric_rows], width, label="completion block acc")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("Task metrics by reranker")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(metrics_path, dpi=160)
        plt.close(fig)
        plots.append(metrics_path.name)

    families = sorted({str(row["family"]) for row in family_rows})
    rerankers = [str(row["reranker"]) for row in metric_rows]
    if families and rerankers:
        family_path = out_dir / "family_selection_scores.png"
        width = 0.8 / max(len(rerankers), 1)
        x = list(range(len(families)))
        fig, ax = plt.subplots(figsize=(9, 4))
        for offset, reranker in enumerate(rerankers):
            values = []
            for family in families:
                match = next(
                    (
                        row for row in family_rows
                        if row["family"] == family and row["reranker"] == reranker
                    ),
                    None,
                )
                values.append(float(match["selection_score"]) if match else 0.0)
            positions = [idx - 0.4 + width / 2 + offset * width for idx in x]
            ax.bar(positions, values, width, label=reranker)
        ax.set_xticks(x)
        ax.set_xticklabels(families)
        ax.set_ylabel("selection score")
        ax.set_title("Per-family selection score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(family_path, dpi=160)
        plt.close(fig)
        plots.append(family_path.name)

    return plots


def _eval_nextstep_rows(
    truth: dict[str, str],
    preds: list[dict[str, object]],
    allowed_ids: set[str],
) -> dict[str, float]:
    hits1 = hits3 = hits5 = total = 0
    rr_total = 0.0
    for row in preds:
        example_id = str(row["EXAMPLE_ID"])
        if example_id not in allowed_ids:
            continue
        target = truth.get(example_id)
        if target is None:
            continue
        ranks = [str(row.get(f"RANK_{idx}", "")) for idx in range(1, 6)]
        total += 1
        hits1 += int(bool(ranks[:1]) and ranks[0] == target)
        hits3 += int(target in ranks[:3])
        hits5 += int(target in ranks[:5])
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


def _eval_completion_rows(
    truth: dict[str, list[str]],
    preds: list[dict[str, object]],
    allowed_ids: set[str],
) -> dict[str, float]:
    total = exact = 0
    edit_total = token_acc_total = block_acc_total = 0.0
    for row in preds:
        example_id = str(row["EXAMPLE_ID"])
        if example_id not in allowed_ids:
            continue
        target = truth.get(example_id)
        if target is None:
            continue
        pred = split_steps(str(row.get("PREDICTED_SEQUENCE", "")))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and checkpoint rerankers on the local dev set.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--dev-dir", type=Path, default=PROJECT_ROOT / "data" / "dev")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "reranker_compare")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=_default_checkpoints())
    parser.add_argument("--transformer-device", default="cpu")
    parser.add_argument("--completion-mode", choices=["prefix", "retrieval", "beam", "ensemble"], default="ensemble")
    parser.add_argument(
        "--selection-scope",
        choices=["all", "checkpoints"],
        default="all",
        help="Select the best overall reranker or require the winner to be an eligible checkpoint reranker.",
    )
    parser.add_argument(
        "--require-selected-checkpoint",
        action="store_true",
        help="Fail after writing metrics if the selected reranker is not a loadable checkpoint.",
    )
    parser.add_argument(
        "--require-checkpoints-available",
        action="store_true",
        help="Fail if any requested checkpoint is missing or cannot load a transformer scorer.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    records = load_corpus(args.data_dir, args.generated_dir)
    corpus_family_counts = Counter(record.family for record in records)
    ranker = NGramRanker(max_order=8).fit(records)
    input_rows = read_rows(args.dev_dir / "eval_input_valid.csv")
    if args.max_examples > 0:
        input_rows = input_rows[:args.max_examples]
    ids_by_family: dict[str, set[str]] = {}
    for row in input_rows:
        ids_by_family.setdefault(row["FAMILY"].strip().upper(), set()).add(row["EXAMPLE_ID"])
    nextstep_truth = {row["EXAMPLE_ID"]: row["NEXT_STEP"] for row in read_rows(args.dev_dir / "nextstep_truth.csv")}
    completion_truth = {
        row["EXAMPLE_ID"]: split_steps(row["REMAINING_SEQUENCE"])
        for row in read_rows(args.dev_dir / "completion_truth.csv")
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing_checkpoints = [path for path in args.checkpoints if not path.exists()]
    if args.require_checkpoints_available and missing_checkpoints:
        for path in missing_checkpoints:
            print(f"Required checkpoint does not exist: {path}")
        raise SystemExit(2)
    candidates: list[Path | None] = [None]
    candidates.extend(path for path in args.checkpoints if path.exists())

    metric_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    print(
        "reranker,available,selection_eligible,nextstep_mrr,nextstep_top1,"
        "completion_exact,completion_token_accuracy,selection_score"
    )
    for checkpoint_path in candidates:
        label = _checkpoint_label(checkpoint_path)
        run_dir = args.out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        engine = CompletionEngine(
            records,
            ranker,
            data_dir=args.data_dir,
            checkpoint_path=checkpoint_path,
            transformer_device=args.transformer_device,
        )

        next_rows: list[dict[str, object]] = []
        completion_rows: list[dict[str, object]] = []
        for row in input_rows:
            family = row["FAMILY"].strip().upper()
            partial = split_steps(row["PARTIAL_SEQUENCE"])
            ranks = ranker.rank_next(family, partial, k=12)
            ranks = engine.rerank_next(family, partial, ranks, k=5)
            ranks = ranks + [""] * (5 - len(ranks))
            next_rows.append({
                "EXAMPLE_ID": row["EXAMPLE_ID"],
                "RANK_1": ranks[0],
                "RANK_2": ranks[1],
                "RANK_3": ranks[2],
                "RANK_4": ranks[3],
                "RANK_5": ranks[4],
            })
            suffix = engine.complete(
                family,
                partial,
                completion_fraction=_fraction(row),
                mode=args.completion_mode,
            )
            completion_rows.append({"EXAMPLE_ID": row["EXAMPLE_ID"], "PREDICTED_SEQUENCE": "|".join(suffix)})

        next_path = run_dir / "nextstep.csv"
        completion_path = run_dir / "completion.csv"
        write_rows(next_path, ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], next_rows)
        write_rows(completion_path, ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], completion_rows)
        nextstep_metrics = eval_nextstep(args.dev_dir / "nextstep_truth.csv", next_path)
        completion_metrics = eval_completion(args.dev_dir / "completion_truth.csv", completion_path)
        score = _selection_score(nextstep_metrics, completion_metrics)
        checkpoint_available = bool(engine.scorer and engine.scorer.available) if checkpoint_path else True
        selection_eligible = checkpoint_available and (
            checkpoint_path is not None or args.selection_scope == "all"
        )
        row = {
            "reranker": label,
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path else "",
            "available": checkpoint_available,
            "selection_eligible": selection_eligible,
            "nextstep_count": nextstep_metrics["count"],
            "nextstep_top1": nextstep_metrics["top1"],
            "nextstep_top3": nextstep_metrics["top3"],
            "nextstep_top5": nextstep_metrics["top5"],
            "nextstep_mrr": nextstep_metrics["mrr"],
            "completion_count": completion_metrics["count"],
            "completion_exact_match": completion_metrics["exact_match"],
            "completion_normalized_edit_distance": completion_metrics["normalized_edit_distance"],
            "completion_token_accuracy": completion_metrics["token_accuracy"],
            "completion_block_accuracy": completion_metrics["block_accuracy"],
            "selection_score": score,
            "selection_scope": args.selection_scope,
            "nextstep_path": str(next_path),
            "completion_path": str(completion_path),
        }
        metric_rows.append(row)
        for family, family_ids in sorted(ids_by_family.items()):
            family_nextstep = _eval_nextstep_rows(nextstep_truth, next_rows, family_ids)
            family_completion = _eval_completion_rows(completion_truth, completion_rows, family_ids)
            family_rows.append({
                "reranker": label,
                "family": family,
                "nextstep_count": family_nextstep["count"],
                "nextstep_top1": family_nextstep["top1"],
                "nextstep_top3": family_nextstep["top3"],
                "nextstep_top5": family_nextstep["top5"],
                "nextstep_mrr": family_nextstep["mrr"],
                "completion_count": family_completion["count"],
                "completion_exact_match": family_completion["exact_match"],
                "completion_normalized_edit_distance": family_completion["normalized_edit_distance"],
                "completion_token_accuracy": family_completion["token_accuracy"],
                "completion_block_accuracy": family_completion["block_accuracy"],
                "selection_score": _selection_score(family_nextstep, family_completion),
            })
        print(
            f"{label},{row['available']},{row['selection_eligible']},"
            f"{nextstep_metrics['mrr']:.4f},{nextstep_metrics['top1']:.4f},"
            f"{completion_metrics['exact_match']:.4f},{completion_metrics['token_accuracy']:.4f},{score:.4f}"
        )

    metric_rows.sort(
        key=lambda item: (bool(item.get("selection_eligible")), float(item["selection_score"])),
        reverse=True,
    )
    fieldnames = [
        "reranker",
        "checkpoint",
        "checkpoint_sha256",
        "available",
        "selection_eligible",
        "nextstep_count",
        "nextstep_top1",
        "nextstep_top3",
        "nextstep_top5",
        "nextstep_mrr",
        "completion_count",
        "completion_exact_match",
        "completion_normalized_edit_distance",
        "completion_token_accuracy",
        "completion_block_accuracy",
        "selection_score",
        "selection_scope",
        "nextstep_path",
        "completion_path",
    ]
    write_rows(args.out_dir / "metrics.csv", fieldnames, metric_rows)
    family_fieldnames = [
        "reranker",
        "family",
        "nextstep_count",
        "nextstep_top1",
        "nextstep_top3",
        "nextstep_top5",
        "nextstep_mrr",
        "completion_count",
        "completion_exact_match",
        "completion_normalized_edit_distance",
        "completion_token_accuracy",
        "completion_block_accuracy",
        "selection_score",
    ]
    write_rows(args.out_dir / "family_metrics.csv", family_fieldnames, family_rows)
    plots = _write_plots(args.out_dir, metric_rows, family_rows)
    _write_report(args.out_dir, metric_rows, family_rows, plots)
    best_row = metric_rows[0] if metric_rows else None
    best_checkpoint = str(best_row["checkpoint"]) if best_row else ""
    summary = {
        "data_dir": str(args.data_dir),
        "generated_dir": str(args.generated_dir),
        "dev_dir": str(args.dev_dir),
        "num_corpus_sequences": len(records),
        "corpus_fingerprint": corpus_fingerprint(records),
        "corpus_family_counts": dict(sorted(corpus_family_counts.items())),
        "completion_mode": args.completion_mode,
        "transformer_device": args.transformer_device,
        "selection_scope": args.selection_scope,
        "require_checkpoints_available": args.require_checkpoints_available,
        "requested_checkpoints": [str(path) for path in args.checkpoints],
        "best_reranker": best_row["reranker"] if best_row else None,
        "best_checkpoint": best_checkpoint,
        "runs": metric_rows,
        "families": family_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "best_checkpoint.txt").write_text(best_checkpoint + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'metrics.csv'}")
    print(f"Wrote {args.out_dir / 'family_metrics.csv'}")
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    print(f"Wrote {args.out_dir / 'best_checkpoint.txt'}")
    print(f"Wrote {args.out_dir / 'REPORT.md'}")
    for plot in plots:
        print(f"Wrote {args.out_dir / plot}")
    if args.require_selected_checkpoint:
        if best_row is None:
            print("Strict reranker selection failed: no reranker rows were produced")
            raise SystemExit(2)
        if not best_row.get("checkpoint"):
            print("Strict reranker selection failed: selected reranker has no checkpoint")
            raise SystemExit(2)
        if not bool(best_row.get("selection_eligible")):
            print("Strict reranker selection failed: selected reranker is not selection eligible")
            raise SystemExit(2)
        print(f"Strict reranker selection passed: {best_row['reranker']}")
    if args.require_checkpoints_available:
        unavailable = [
            row for row in metric_rows
            if row.get("checkpoint") and not bool(row.get("available"))
        ]
        if unavailable:
            print("Strict checkpoint availability failed:")
            for row in unavailable:
                print(f"- {row['reranker']} scorer unavailable for {row['checkpoint']}")
            raise SystemExit(2)
        print("Strict checkpoint availability passed")


if __name__ == "__main__":
    main()
