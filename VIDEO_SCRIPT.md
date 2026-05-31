# Demo Video Script

Target length: 2 minutes maximum. Format: MP4, 1080p, with audio.

## Tally description

Two-minute walkthrough by Coding Club, Evgenii Avdeev and Swathi Dutt, for Track 1: Learning and Benchmarking Process Logic. The demo presents the current submitted result: a reproducible Industrial AI Process Logic pipeline covering next-step prediction, sequence completion, and anomaly detection, with local smoke evidence, packaging verification, and a clear boundary that no final Leonardo-trained checkpoint or official score is claimed. Repository: https://github.com/j-avdeev/industrial-ai-process-logic.

## Recording plan

0:00-0:10 - Slide 1: Team and track

Show `SLIDES.pdf` slide 1.

Voiceover:
"We are Coding Club: Evgenii Avdeev and Swathi Dutt. This is Track 1, Learning and Benchmarking Process Logic."

0:10-0:25 - Slides 2-3: Project and problem

Show slides 2 and 3, then briefly show `README.md`.

Voiceover:
"Our submission is Industrial AI Process Logic. The goal is to model long semiconductor process routes, predict plausible next steps, complete partial routes, and detect invalid sequences across MOSFET, IGBT, and IC families."

0:25-0:45 - Slides 4-5: Approach and local task coverage

Show slides 4 and 5, plus `industrial_ai/infer.py`.

Voiceover:
"The implementation treats full process steps as tokens, conditions on product family, and uses official grammar and validator logic as a reliability layer. The local inference path implements all three submitted tasks: nextstep, completion, and anomaly."

0:45-1:10 - Live smoke evidence

Run:

```powershell
cd C:\Users\j-avd\Documents\Hackathon29.05\industrial-ai-process-logic

python -m industrial_ai.make_devset `
  --out-dir artifacts/video_demo/dev `
  --valid-per-family 2 `
  --anomaly-valid-per-family 2 `
  --anomaly-invalid-per-family 2

python -m industrial_ai.infer `
  --valid-input artifacts/video_demo/dev/eval_input_valid.csv `
  --anomaly-input artifacts/video_demo/dev/eval_input_anomaly.csv `
  --completion-mode ensemble `
  --out-dir artifacts/video_demo/submissions

python -m industrial_ai.metrics --dev-dir artifacts/video_demo/dev --pred-dir artifacts/video_demo/submissions
```

Voiceover:
"This creates a small dev set, generates all three submission CSVs, and prints local metrics. Next-step has a working baseline, anomaly detection is strong on smoke data, and completion runs end to end even though exact match is weak on tiny data."

1:10-1:28 - Slides 6-7: Prepared Leonardo path and verification

Show slides 6 and 7, then `LEONARDO_RUNBOOK.md`.

Voiceover:
"The repository includes a prepared Leonardo path: source bundling, launch scripts, checkpoint audits, package creation, and returned-package verification. The verification code binds hashes, run manifests, checkpoint evidence, and package contents so the final artifacts can be audited."

1:28-1:45 - Slide 8: Task completion status

Show slide 8 and `REPORT.md`.

Voiceover:
"The current result is a completed reproducible pipeline for Track 1 Tasks 1 through 3. I do not claim a final official score, completed Leonardo return package, or final selected checkpoint in the current evidence."

1:45-2:00 - Slide 9: Current submission result

Show slide 9 and the GitHub repository URL.

Voiceover:
"The submitted result is a public MIT repository with README, report, dependency manifest, slides, and this demo script. Technically, all three task paths run locally and produce submission CSVs; final benchmark-quality training is outside the claimed result."
