# REPORT

## TL;DR

This repository implements a reproducible Industrial AI pipeline for semiconductor process sequences. It combines a grammar-aware baseline, official validator-based anomaly detection, optional step-token transformer training, and exact submission CSV generation for the three Track 1 tasks.

## Problem

The challenge asks whether models can learn process logic in long semiconductor fabrication routes rather than only memorizing common step strings. The scored tasks are next-step prediction, sequence completion, and anomaly detection.

## Approach

- Treat each process step string as one token.
- Condition predictions on product family: `MOSFET`, `IGBT`, or `IC`.
- Use the official process grammar and validator as a strong reliability layer.
- Use n-gram and nearest-prefix baselines for immediate reproducible predictions.
- Use deterministic validator output for anomaly detection and rule attribution.
- Improve completion as candidate generation plus reranking: retrieval from a larger generated corpus, n-gram beam fallback, official validation, length penalties, and optional transformer checkpoint scoring.
- Provide optional PyTorch transformer training for scaling experiments across `tiny`, `small`, and `medium` configurations.

## Current Results

The repo is ready to run local synthetic dev evaluation:

```bash
python -m industrial_ai.prepare
python -m industrial_ai.make_devset --valid-per-family 5 --anomaly-valid-per-family 5 --anomaly-invalid-per-family 5
python -m industrial_ai.infer --completion-mode ensemble
python -m industrial_ai.metrics
python -m industrial_ai.compare_completion
```

Official leaderboard metrics require the organizer-provided eval files and ground-truth scoring script. The final CSV outputs are produced under `submissions/`.

## How To Run

See `README.md` for setup, local smoke run, official eval inference, neural training, and Leonardo Slurm commands.

## What Worked

- The official validator gives a robust anomaly detector for known process-rule violations.
- Long-format sequence data is simple to convert into family-conditioned token sequences.
- A prefix index is a strong completion baseline when generated corpora contain close route variants.
- Larger generated corpora should improve completion because the retrieval/reranking path has more valid suffix candidates.

## What Did Not Work Yet

- Official eval ground truth is not available in this repo, so real final scores cannot be computed locally.
- Transformer checkpoints are not committed yet; they should be trained on Leonardo and referenced here after runs finish.
- The current neural model is intentionally compact and should be tuned with generated data volume, model size, and epoch sweeps.

## Next 36-Hour Improvements

- Generate 30k to 150k additional valid sequences and rerun scaling experiments.
- Train `tiny`, `small`, and `medium` models on Leonardo and compare metrics.
- Add loss curves and per-family breakdown plots to `artifacts/`.
- Use transformer logits to re-rank grammar-valid candidates for next-step prediction.
- Record a short baseline-vs-trained demo using identical inputs.

## Credits & Dependencies

- Official challenge data: Lumos Data / Zero One Hack Track 1 Industrial AI.
- Core implementation: Python stdlib baseline and optional PyTorch model.
- Compute target: CINECA Leonardo GPU cluster.
