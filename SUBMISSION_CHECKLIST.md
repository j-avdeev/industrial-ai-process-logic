# Submission Checklist

Based on the Zero One submission requirements at https://docs.zero-one.lumos-consulting.at/tracks/submissions/.

## Repository

- [x] Public repository URL is reachable: `https://github.com/j-avdeev/industrial-ai-process-logic`.
- [x] Root `LICENSE` uses MIT license text.
- [x] Root `README.md` includes setup and run instructions.
- [x] Root `REPORT.md` describes problem, approach, results, gaps, and next steps.
- [x] Root dependency manifest exists: `requirements.txt`.
- [x] `.env` is ignored and only `.env.example` is tracked.
- [x] Clean-checkout commands are documented.
- [ ] Confirm no secrets are present in git history before final form submission.

## Slides and Video

- [x] Slide source exists: `SLIDES.md`.
- [x] Upload-ready PDF deck exists: `SLIDES.pdf`.
- [x] Demo recording plan and Tally description exist: `VIDEO_SCRIPT.md`.
- [ ] Record and upload/link an MP4 demo under 2 minutes.

## Track 1 Deliverables

- [x] Code path writes `submissions/nextstep.csv`.
- [x] Code path writes `submissions/completion.csv`.
- [x] Code path writes `submissions/anomaly.csv`.
- [x] Packaging code writes `artifacts/submission_package/track1_submission.zip`.
- [x] Local smoke path validates inference, metrics, package manifest, and verification plumbing.
- [x] Leonardo scripts cover 50k-150k generated sequences per family and tiny/small/medium training.
- [x] Returned-package verifier requires final Leonardo evidence before final success is claimed.
- [ ] Stage official eval CSVs in `data/eval/`.
- [ ] Run the final Leonardo generation/training/reranker pipeline.
- [ ] Copy back `artifacts/leonardo_return_packet.zip` and verify it.
- [ ] Upload final official CSV package and evidence after returned-package verification passes.

## Final Local Checks

```bash
python -m py_compile industrial_ai/*.py
python -m industrial_ai.preflight
python -m industrial_ai.leonardo_shell_audit
python -m industrial_ai.smoke_pipeline --out-dir artifacts/submission_smoke --valid-per-family 2
```
