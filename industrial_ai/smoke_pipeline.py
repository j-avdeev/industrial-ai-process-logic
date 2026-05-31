from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .paths import PROJECT_ROOT


def _run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def _python_module(module: str, *args: str | Path) -> list[str]:
    return [sys.executable, "-m", module, *(str(arg) for arg in args)]


def _package_args(
    submissions_dir: Path,
    out_dir: Path,
    checkpoint_dir: Path,
    generated_dir: Path,
    require_evidence: bool,
    generated_per_family: int,
    min_reranker_count: int,
    min_completion_compare_count: int,
    min_train_epochs: int,
    require_selected_checkpoint: bool,
    transformer_device: str,
    required_manifest_stage: str = "",
) -> list[str | Path]:
    args: list[str | Path] = [
        "industrial_ai.package_submission",
        "--submission-dir", submissions_dir,
        "--artifacts-dir", out_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--generated-dir", generated_dir,
        "--out-dir", out_dir / "submission_package",
        "--include-evidence",
    ]
    if require_evidence:
        args.extend([
            "--require-evidence",
            "--required-min-generated-per-family", str(generated_per_family),
            "--required-min-reranker-count", str(min_reranker_count),
            "--required-min-completion-compare-count", str(min_completion_compare_count),
            "--required-min-train-epochs", str(min_train_epochs),
            "--required-checkpoint-sizes", "tiny",
        ])
        if generated_per_family > 0:
            args.extend([
                "--required-max-generated-per-family", str(generated_per_family),
                "--require-generated-metadata",
            ])
        if require_selected_checkpoint:
            args.extend([
                "--require-selected-checkpoint",
                "--required-transformer-device", transformer_device,
            ])
        args.append("--require-preflight-eval")
        if required_manifest_stage:
            args.extend(["--required-manifest-stage", required_manifest_stage])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cheap local end-to-end smoke pipeline.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "local_smoke")
    parser.add_argument("--with-training", action="store_true", help="Train a one-epoch tiny CPU checkpoint and validate it.")
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--valid-per-family", type=int, default=5)
    parser.add_argument(
        "--require-selected-checkpoint",
        action="store_true",
        help="Require checkpoint-only reranker selection and strict inference/package evidence.",
    )
    parser.add_argument(
        "--generated-per-family",
        type=int,
        default=0,
        help="Generate this many extra sequences per family inside the smoke out-dir and require their metadata.",
    )
    args = parser.parse_args()
    if args.generated_per_family < 0:
        raise SystemExit("--generated-per-family must be non-negative")
    if args.require_selected_checkpoint and not args.with_training:
        raise SystemExit("--require-selected-checkpoint requires --with-training")

    out_dir = args.out_dir
    dev_dir = out_dir / "dev"
    generated_dir = out_dir / ("generated" if args.generated_per_family else "generated_disabled")
    checkpoint_dir = out_dir / "checkpoints"
    submissions_dir = out_dir / "submissions"
    checkpoint_path = checkpoint_dir / "tiny" / "model.pt"
    missing_checkpoint_path = out_dir / "missing_checkpoint.pt"
    comparison_examples = args.valid_per_family * 6

    _run(_python_module("industrial_ai.leonardo_shell_audit", "--out", out_dir / "leonardo_shell_audit.json"))
    _run(_python_module("industrial_ai.preflight", "--out", out_dir / "preflight_initial.json"))
    if args.generated_per_family > 0:
        _run(_python_module(
            "industrial_ai.generate_extra",
            "--family", "mosfet",
            "--count", str(args.generated_per_family),
            "--seed", "101",
            "--output", generated_dir / "MOSFET_extra.csv",
            "--skip-if-complete",
            "--exact-count",
        ))
        _run(_python_module(
            "industrial_ai.generate_extra",
            "--family", "igbt",
            "--count", str(args.generated_per_family),
            "--seed", "102",
            "--output", generated_dir / "IGBT_extra.csv",
            "--skip-if-complete",
            "--exact-count",
        ))
        _run(_python_module(
            "industrial_ai.generate_extra",
            "--family", "ic",
            "--count", str(args.generated_per_family),
            "--seed", "103",
            "--output", generated_dir / "IC_extra.csv",
            "--skip-if-complete",
            "--exact-count",
        ))
    _run(_python_module(
        "industrial_ai.audit_corpus",
        "--generated-dir", generated_dir,
        "--min-generated-per-family", str(args.generated_per_family),
        "--max-generated-per-family", str(args.generated_per_family),
        "--out-dir", out_dir / "corpus_audit",
    ))
    _run(_python_module("industrial_ai.prepare", "--generated-dir", generated_dir, "--out-dir", out_dir / "prepared"))
    _run(_python_module(
        "industrial_ai.make_devset",
        "--out-dir", dev_dir,
        "--valid-per-family", args.valid_per_family,
        "--anomaly-valid-per-family", args.valid_per_family,
        "--anomaly-invalid-per-family", args.valid_per_family,
    ))
    _run(_python_module(
        "industrial_ai.preflight",
        "--valid-input", dev_dir / "eval_input_valid.csv",
        "--anomaly-input", dev_dir / "eval_input_anomaly.csv",
        "--require-eval",
        "--out", out_dir / "preflight_full_pipeline.json",
    ))

    if args.with_training:
        _run(_python_module(
            "industrial_ai.train",
            "--model-size", "tiny",
            "--epochs", str(args.train_epochs),
            "--batch-size", str(args.train_batch_size),
            "--device", "cpu",
            "--generated-dir", generated_dir,
            "--out-dir", checkpoint_dir,
            "--skip-if-complete",
        ))
        _run(_python_module(
            "industrial_ai.checkpoint_audit",
            "--artifacts-dir", out_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--model-sizes", "tiny",
            "--min-generated-per-family", str(args.generated_per_family),
            "--max-generated-per-family", str(args.generated_per_family),
            "--require-checkpoint-device", "cpu",
            "--min-train-epochs", str(args.train_epochs),
        ))
        _run(_python_module(
            "industrial_ai.run_manifest",
            "--out", out_dir / "run_manifest.json",
            "--artifacts-dir", out_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--submission-dir", submissions_dir,
            "--stage", "local_smoke_checkpoint_audited",
            "--set", f"WITH_TRAINING={args.with_training}",
            "--set", f"VALID_PER_FAMILY={args.valid_per_family}",
            "--set", f"GENERATED_PER_FAMILY={args.generated_per_family}",
            "--set", f"COUNT_PER_FAMILY={args.generated_per_family}",
            "--set", f"REQUIRE_SELECTED_CHECKPOINT={args.require_selected_checkpoint}",
        ))

    completion_compare_args: list[str | Path] = [
        "--dev-dir", dev_dir,
        "--generated-dir", generated_dir,
        "--out-dir", out_dir / "completion_compare",
        "--checkpoint", checkpoint_path if args.with_training else missing_checkpoint_path,
        "--max-examples", str(comparison_examples),
    ]
    if args.with_training:
        completion_compare_args.extend(["--transformer-device", "cpu"])
        if args.require_selected_checkpoint:
            completion_compare_args.extend(["--require-checkpoint", "--require-transformer-available"])
    _run(_python_module("industrial_ai.compare_completion", *completion_compare_args))

    reranker_args: list[str | Path] = [
        "--dev-dir", dev_dir,
        "--generated-dir", generated_dir,
        "--out-dir", out_dir / "reranker_compare",
    ]
    if args.with_training:
        reranker_args.extend(["--checkpoints", checkpoint_path])
        reranker_args.extend(["--transformer-device", "cpu"])
        if args.require_selected_checkpoint:
            reranker_args.extend([
                "--selection-scope", "checkpoints",
                "--require-selected-checkpoint",
                "--require-checkpoints-available",
            ])
    else:
        reranker_args.extend(["--checkpoints", missing_checkpoint_path])
    _run(_python_module("industrial_ai.compare_rerankers", *reranker_args))

    infer_args: list[str | Path] = [
        "--valid-input", dev_dir / "eval_input_valid.csv",
        "--anomaly-input", dev_dir / "eval_input_anomaly.csv",
        "--completion-mode", "ensemble",
        "--generated-dir", generated_dir,
        "--checkpoint", checkpoint_path if args.with_training else missing_checkpoint_path,
        "--out-dir", submissions_dir,
    ]
    if args.with_training:
        infer_args.extend(["--transformer-device", "cpu"])
        if args.require_selected_checkpoint:
            infer_args.extend([
                "--reranker-metrics", out_dir / "reranker_compare" / "metrics.json",
                "--require-checkpoint",
                "--require-transformer-available",
                "--require-selected-checkpoint",
            ])
    _run(_python_module("industrial_ai.infer", *infer_args))
    _run(_python_module("industrial_ai.metrics", "--dev-dir", dev_dir, "--pred-dir", submissions_dir))

    _run(_python_module(
        "industrial_ai.run_manifest",
        "--out", out_dir / "run_manifest.json",
        "--artifacts-dir", out_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--submission-dir", submissions_dir,
        "--stage", "local_smoke",
        "--set", f"WITH_TRAINING={args.with_training}",
        "--set", f"VALID_PER_FAMILY={args.valid_per_family}",
        "--set", f"GENERATED_PER_FAMILY={args.generated_per_family}",
        "--set", f"COUNT_PER_FAMILY={args.generated_per_family}",
        "--set", f"REQUIRE_SELECTED_CHECKPOINT={args.require_selected_checkpoint}",
    ))

    if args.with_training:
        _run(_python_module(
            "industrial_ai.validate_run",
            "--artifacts-dir", out_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--submission-dir", submissions_dir,
            "--model-sizes", "tiny",
            "--min-generated-per-family", str(args.generated_per_family),
            "--max-generated-per-family", str(args.generated_per_family),
            "--min-reranker-count", str(args.valid_per_family * 6),
            "--min-completion-compare-count", str(comparison_examples),
            "--min-train-epochs", str(args.train_epochs),
            "--require-preflight",
            "--require-preflight-eval",
            "--require-submissions",
        ))
        if args.require_selected_checkpoint:
            validation_args = [
                "industrial_ai.validate_run",
                "--artifacts-dir", out_dir,
                "--checkpoint-dir", checkpoint_dir,
                "--submission-dir", submissions_dir,
                "--model-sizes", "tiny",
                "--min-generated-per-family", str(args.generated_per_family),
                "--max-generated-per-family", str(args.generated_per_family),
                "--min-reranker-count", str(args.valid_per_family * 6),
                "--min-completion-compare-count", str(comparison_examples),
                "--min-train-epochs", str(args.train_epochs),
                "--require-preflight",
                "--require-preflight-eval",
                "--require-transformer-device", "cpu",
                "--require-selected-checkpoint",
                "--require-submissions",
            ]
            _run(_python_module(*validation_args))
        _run(_python_module(
            "industrial_ai.run_manifest",
            "--out", out_dir / "run_manifest.json",
            "--artifacts-dir", out_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--submission-dir", submissions_dir,
            "--stage", "local_smoke_validated",
            "--set", f"WITH_TRAINING={args.with_training}",
            "--set", f"VALID_PER_FAMILY={args.valid_per_family}",
            "--set", f"GENERATED_PER_FAMILY={args.generated_per_family}",
            "--set", f"COUNT_PER_FAMILY={args.generated_per_family}",
            "--set", f"REQUIRE_SELECTED_CHECKPOINT={args.require_selected_checkpoint}",
        ))

    _run(_python_module(*_package_args(
        submissions_dir,
        out_dir,
        checkpoint_dir,
        generated_dir,
        args.with_training,
        args.generated_per_family,
        args.valid_per_family * 6,
        comparison_examples,
        args.train_epochs,
        args.require_selected_checkpoint,
        "cpu" if args.with_training else "",
    )))
    _run(_python_module(
        "industrial_ai.run_manifest",
        "--out", out_dir / "run_manifest.json",
        "--artifacts-dir", out_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--submission-dir", submissions_dir,
        "--stage", "local_smoke_packaged",
        "--set", f"WITH_TRAINING={args.with_training}",
        "--set", f"VALID_PER_FAMILY={args.valid_per_family}",
        "--set", f"GENERATED_PER_FAMILY={args.generated_per_family}",
        "--set", f"COUNT_PER_FAMILY={args.generated_per_family}",
        "--set", f"REQUIRE_SELECTED_CHECKPOINT={args.require_selected_checkpoint}",
    ))
    _run(_python_module(*_package_args(
        submissions_dir,
        out_dir,
        checkpoint_dir,
        generated_dir,
        args.with_training,
        args.generated_per_family,
        args.valid_per_family * 6,
        comparison_examples,
        args.train_epochs,
        args.require_selected_checkpoint,
        "cpu" if args.with_training else "",
        "local_smoke_packaged",
    )))
    _run(_python_module(
        "industrial_ai.verify_package",
        "--package-dir", out_dir / "submission_package",
    ))
    if args.with_training:
        final_audit_args: list[str | Path] = [
            "industrial_ai.final_audit",
            "--artifacts-dir", out_dir,
            "--package-dir", out_dir / "submission_package",
            "--required-manifest-stage", "local_smoke_packaged",
            "--required-checkpoint-sizes", "tiny",
            "--min-generated-per-family", str(args.generated_per_family),
            "--max-generated-per-family", str(args.generated_per_family),
            "--min-reranker-count", str(args.valid_per_family * 6),
            "--min-completion-compare-count", str(comparison_examples),
            "--min-train-epochs", str(args.train_epochs),
            "--required-transformer-device", "cpu" if args.require_selected_checkpoint else "",
            "--require-selected-checkpoint" if args.require_selected_checkpoint else "--no-require-selected-checkpoint",
            "--no-require-preflight-cuda",
            "--require-preflight-eval",
            "--no-require-readiness",
            "--no-require-source-bundle-proof",
        ]
        final_audit_args.append(
            "--require-generated-metadata"
            if args.generated_per_family > 0
            else "--no-require-generated-metadata"
        )
        _run(_python_module(*final_audit_args))
        returned_package_args: list[str | Path] = [
            "industrial_ai.verify_returned_package",
            "--artifacts-dir", out_dir,
            "--package-dir", out_dir / "submission_package",
            "--required-manifest-stage", "local_smoke_packaged",
            "--required-checkpoint-sizes", "tiny",
            "--min-generated-per-family", str(args.generated_per_family),
            "--max-generated-per-family", str(args.generated_per_family),
            "--min-reranker-count", str(args.valid_per_family * 6),
            "--min-completion-compare-count", str(comparison_examples),
            "--min-train-epochs", str(args.train_epochs),
            "--required-transformer-device", "cpu" if args.require_selected_checkpoint else "",
            "--require-selected-checkpoint" if args.require_selected_checkpoint else "--no-require-selected-checkpoint",
            "--no-require-preflight-cuda",
            "--require-preflight-eval",
            "--no-require-readiness",
            "--no-require-source-bundle-proof",
        ]
        returned_package_args.append(
            "--require-generated-metadata"
            if args.generated_per_family > 0
            else "--no-require-generated-metadata"
        )
        _run(_python_module(*returned_package_args))
    else:
        print("Skipping final audit; rerun with --with-training to exercise strict evidence checks")
    print(f"Smoke pipeline complete: {out_dir}")


if __name__ == "__main__":
    main()
