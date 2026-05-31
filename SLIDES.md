# Submission Slides

## Slide 1: Team and Track

- Team name: Coding Club.
- Team members: Evgenii Avdeev, Swathi Dutt.
- Track 1: Learning and Benchmarking Process Logic.

## Slide 2: Industrial AI Process Logic

- Repository: https://github.com/j-avdeev/industrial-ai-process-logic.
- Track 1 pipeline for semiconductor process next-step, completion, and anomaly tasks.
- Built as a reproducible baseline, training, Leonardo, and packaging workflow.

## Slide 3: Problem

- Process routes are long, family-specific, and constrained by hard fabrication rules.
- A useful system must predict plausible next steps, complete partial routes, and flag invalid sequences.
- The jury needs runnable code, honest metrics, and evidence that the final artifacts match the run.

## Slide 4: Core Approach

- Treat full process steps as tokens and condition every prediction on MOSFET, IGBT, or IC family.
- Use official grammar and validator logic as a reliability layer for anomalies and candidate filtering.
- Combine retrieval, n-gram beam search, validator penalties, and optional transformer checkpoint reranking.

## Slide 5: What Runs Locally

- `industrial_ai.smoke_pipeline` creates dev inputs, predictions, metrics, package manifests, and audits.
- `industrial_ai.infer` implements all three submitted tasks: nextstep, completion, and anomaly.
- Local smoke checks are intentionally small; they prove the pipeline, not final leaderboard quality.

## Slide 6: Leonardo Path

- Generate 50k-150k valid sequences per product family with the official generator.
- Train tiny, small, and medium transformer checkpoints on CUDA.
- Select a checkpoint reranker, run official inference, and package submissions plus evidence.

## Slide 7: Verification

- Preflight validates raw data, eval staging, PyTorch, CUDA, and expected input schemas.
- Corpus, checkpoint, validation, final-audit, and returned-package checks bind hashes and thresholds.
- Source-bundle proof prevents script or Python source drift between upload, training, and packaging.

## Slide 8: Task Completion Status

- Completed: reproducible pipeline covering Track 1 Tasks 1-3.
- Local smoke evidence: next-step works, anomaly detection is strong, completion path runs but exact match is weak on tiny data.
- Pending: final official eval outputs, Leonardo training run, returned package, and final checkpoint claim.

## Slide 9: Next Step

- Stage official eval CSVs and run the Leonardo readiness command.
- Submit the 50k or 150k-per-family full pipeline with CUDA training.
- Verify `artifacts/leonardo_return_packet.zip` before uploading the final CSV package.
