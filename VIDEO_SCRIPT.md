# Demo Video Script

Target length: 2 minutes maximum. Format: MP4, 1080p, with audio.

## Tally description

Two-minute walkthrough of Industrial AI Process Logic: a reproducible Track 1 pipeline that generates valid semiconductor process routes, runs next-step/completion/anomaly inference, trains optional transformer checkpoints on Leonardo, selects a checkpoint reranker, and packages auditable submission evidence. Repository: https://github.com/j-avdeev/industrial-ai-process-logic.

## Recording plan

0:00-0:15 - Problem

Show `README.md` and `data/raw/training_data/`.

Voiceover:
"The task is to learn semiconductor process logic. We need to predict next steps, complete partial routes, and detect invalid sequences across MOSFET, IGBT, and IC families without breaking the official process rules."

0:15-0:35 - Pipeline

Show `industrial_ai/` and the local smoke command.

Voiceover:
"The repo is organized as a reproducible pipeline: data prep, official-generator augmentation, grammar-aware inference, optional transformer training, reranker comparison, validation, and package verification."

0:35-1:05 - Live run

Run:

```bash
python -m industrial_ai.make_devset --out-dir artifacts/video_demo/dev --valid-per-family 2 --anomaly-valid-per-family 2 --anomaly-invalid-per-family 2
python -m industrial_ai.infer --valid-input artifacts/video_demo/dev/eval_input_valid.csv --anomaly-input artifacts/video_demo/dev/eval_input_anomaly.csv --completion-mode ensemble --out-dir artifacts/video_demo/submissions
python -m industrial_ai.metrics --dev-dir artifacts/video_demo/dev --pred-dir artifacts/video_demo/submissions
```

Voiceover:
"This creates a small dev set, runs the same inference entry point used for final CSV generation, and prints local metrics. The final official score needs organizer eval files, but this proves the runnable path."

1:05-1:30 - Evidence and packaging

Show `LEONARDO_RUNBOOK.md`, `industrial_ai/package_submission.py`, and `industrial_ai/verify_returned_package.py`.

Voiceover:
"The Leonardo path scales the corpus to 50k to 150k valid sequences per family, trains tiny, small, and medium checkpoints on CUDA, selects a checkpoint reranker, and packages submissions with hashes, run manifests, checkpoint summaries, and audit evidence."

1:30-1:50 - Honest status

Show `REPORT.md` and `SUBMISSION_CHECKLIST.md`.

Voiceover:
"The mature result today is the auditable pipeline and handoff. I am not claiming final Leonardo scores until official eval files are staged and the returned package verifier reports final objective ready."

1:50-2:00 - Close

Show `SLIDES.pdf` and the repository URL.

Voiceover:
"The repo is public, MIT licensed, documented, and ready for the final Leonardo run and submission package."
