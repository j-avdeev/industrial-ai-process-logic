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

## Slide 6: Prepared Leonardo Path

- Repository includes scripts to generate 50k-150k valid sequences per product family.
- Training path covers tiny, small, and medium transformer checkpoints on CUDA.
- Packaging path records hashes, run manifests, checkpoint evidence, and verifier results.

## Slide 7: Verification

- Preflight validates raw data, eval staging, PyTorch, CUDA, and expected input schemas.
- Corpus, checkpoint, validation, final-audit, and returned-package checks bind hashes and thresholds.
- Source-bundle proof prevents script or Python source drift between upload, training, and packaging.

## Slide 8: Task Completion Status

- Completed: reproducible pipeline covering Track 1 Tasks 1-3.
- Local smoke evidence: next-step works, anomaly detection is strong, completion path runs but exact match is weak on tiny data.
- Not claimed: final official score, completed Leonardo return package, or final selected checkpoint.

## Slide 9: Current Submission Result

- Submitted result: public MIT repo with README, REPORT, dependency manifest, slides, and demo script.
- Technical result: all three Track 1 task paths run locally and produce submission CSVs.
- Boundary: final benchmark-quality checkpoint and official scores are outside the current submitted evidence.
