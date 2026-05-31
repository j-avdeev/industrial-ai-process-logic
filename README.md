# Industrial AI Process Logic

Zero One Hack Track 1 solution for learning and benchmarking semiconductor process logic.

**Submission repository:** [https://github.com/j-avdeev/industrial-ai-process-logic](https://github.com/j-avdeev/industrial-ai-process-logic)

The repository is built around the official Industrial AI starter pack from `Lumos-Data/zero_one_hack_01`. It includes the public training data, a grammar-aware baseline, a deterministic anomaly detector using the official validator, optional PyTorch transformer training, local development metrics, and Leonardo Slurm scripts.

## What Is Included

- Official copied data in `data/raw/training_data/`
- `industrial_ai.prepare`: vocabulary and corpus stats
- `industrial_ai.preflight`: fail-fast environment, data, eval, and CUDA checks
- `industrial_ai.generate_extra`: synthetic sequence generation through the official generator
- `industrial_ai.audit_corpus`: corpus size and generated-data threshold checks
- `industrial_ai.make_devset`: local dev eval creation
- `industrial_ai.infer`: submission CSV generation
- `industrial_ai.metrics`: local dev scoring
- `industrial_ai.compare_completion`: compare completion strategies
- `industrial_ai.compare_rerankers`: compare baseline vs checkpoint rerankers
- `industrial_ai.checkpoint_audit`: fail-fast checkpoint/corpus consistency audit
- `industrial_ai.run_manifest`: capture run configuration and environment metadata
- `industrial_ai.validate_run`: final artifact readiness checks
- `industrial_ai.package_submission`: ZIP final CSVs with optional evidence files
- `industrial_ai.verify_package`: verify the packaged ZIP and checksum after transfer
- `industrial_ai.verify_returned_package`: run ZIP verification plus final returned-artifact audit
- `industrial_ai.leonardo_return_packet`: package and verify a single Leonardo copy-back ZIP
- `industrial_ai.final_audit`: final pre-submit audit for returned Leonardo artifacts
- `industrial_ai.run_evidence_report`: summarize returned-run evidence against the final Leonardo objective
- `industrial_ai.leonardo_bundle`: build a source bundle for staging the run on Leonardo
- `industrial_ai.leonardo_handoff`: audit the local upload bundle, readiness, launch commands, and self-test evidence
- `industrial_ai.leonardo_shell_audit`: local shell-script sanity check for Leonardo scripts
- `industrial_ai.leonardo_readiness`: launch-readiness checks and sbatch command summary
- `industrial_ai.source_bundle_proof_selftest`: verify source-bundle proof gates reject tampered evidence
- `industrial_ai.smoke_pipeline`: local end-to-end pipeline check
- `industrial_ai.train`: optional step-token transformer training
- Leonardo scripts in `scripts/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The baseline pipeline itself uses only Python stdlib. `torch` is needed only for neural training.

## Local Smoke Run

```bash
python -m industrial_ai.prepare
python -m industrial_ai.make_devset --valid-per-family 5 --anomaly-valid-per-family 5 --anomaly-invalid-per-family 5
python -m industrial_ai.infer
python -m industrial_ai.metrics
```

One command can exercise the same artifact chain used by the Leonardo path:

```bash
python -m industrial_ai.smoke_pipeline --with-training
python -m industrial_ai.smoke_pipeline --with-training --generated-per-family 1
python -m industrial_ai.smoke_pipeline --out-dir artifacts/local_smoke_generated_selected --with-training --generated-per-family 1 --require-selected-checkpoint
```

This creates an isolated run under `artifacts/local_smoke/`, including dev inputs, predictions, metrics, a one-epoch tiny checkpoint, checkpoint audit, reranker comparison, validation, package manifest, ZIP, final audit, returned-package verification, and run evidence report when training is enabled.
Without `--with-training`, the smoke pipeline verifies the basic ZIP package but skips final audit because strict evidence mode requires validation and checkpoint training artifacts.
With `--generated-per-family`, the smoke run generates a tiny isolated corpus under the smoke output directory and requires generated metadata through validation, packaging, final audit, returned-package verification, and evidence reporting. Generated metadata includes the CSV SHA-256, and corpus audit rejects sidecars that do not match the generated file bytes. Add `--require-selected-checkpoint` to mirror the Leonardo final path more closely: compare checkpoint rerankers first, require a checkpoint winner, then run inference with that selected checkpoint and require the selected-checkpoint evidence in the package.
After the strict generated smoke above, the returned-package wrapper can be checked locally with:

```bash
python -m industrial_ai.verify_returned_package --artifacts-dir artifacts/local_smoke_generated_selected --package-dir artifacts/local_smoke_generated_selected/submission_package --required-manifest-stage local_smoke_packaged --required-checkpoint-sizes tiny --min-generated-per-family 1 --max-generated-per-family 1 --min-reranker-count 6 --min-completion-compare-count 0 --min-train-epochs 0 --required-transformer-device cpu --require-selected-checkpoint --no-require-preflight-cuda --require-preflight-eval --require-generated-metadata --no-require-readiness --no-require-source-bundle-proof
```

On these tiny synthetic smoke sets, completion exact match can remain `0.0000`. That is expected at this stage. The useful signal is that the data generation, training, checkpoint reranking, validation, and packaging paths execute end to end. The intended quality path is the larger 50k-150k-per-family generated corpus on Leonardo followed by `tiny`/`small`/`medium` training and reranker selection. Smoke evidence can report `objective_ready=true` for its relaxed local thresholds, but it must also report `final_leonardo_objective_ready=false`.

Run preflight checks directly:

```bash
python -m industrial_ai.preflight
python -m industrial_ai.preflight --require-torch --require-cuda
python -m industrial_ai.leonardo_readiness
```

This creates:

- `data/dev/eval_input_valid.csv`
- `data/dev/eval_input_anomaly.csv`
- `submissions/nextstep.csv`
- `submissions/completion.csv`
- `submissions/anomaly.csv`
- `submissions/inference_summary.json`

## Official Eval Inference

When organizers provide the official eval files, stage them into `data/eval/` with the helper below. It validates required headers and non-empty rows, copies the files to the expected names, and writes `artifacts/eval_staging_manifest.json` with source and staged SHA-256 hashes. Job-time readiness now requires that manifest when `--require-eval` is used without `--defer-eval-staging`, so stale or hand-copied eval files fail before the expensive Leonardo job starts. The repo tracks `data/eval/.gitkeep` so a fresh Leonardo checkout already has the expected drop directory.

```bash
python -m industrial_ai.stage_eval_inputs \
  --valid-input /path/to/official/eval_input_valid.csv \
  --anomaly-input /path/to/official/eval_input_anomaly.csv

python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle
```

```bash
python -m industrial_ai.infer \
  --valid-input data/eval/eval_input_valid.csv \
  --anomaly-input data/eval/eval_input_anomaly.csv \
  --completion-mode ensemble \
  --out-dir submissions
```

Required Track 1 submission files:

- `submissions/nextstep.csv`
- `submissions/completion.csv`
- `submissions/anomaly.csv`

Inference also writes `submissions/inference_summary.json` with row counts, corpus size, corpus fingerprint, completion mode, checkpoint used, checkpoint hash, and transformer availability.

## Training

Generate more valid sequences:

```bash
python -m industrial_ai.generate_extra --family mosfet --count 10000 --seed 101 --output data/generated/MOSFET_extra.csv
python -m industrial_ai.generate_extra --family igbt --count 10000 --seed 102 --output data/generated/IGBT_extra.csv
python -m industrial_ai.generate_extra --family ic --count 10000 --seed 103 --output data/generated/IC_extra.csv
```

For leaderboard completion quality, use at least 10k extra sequences per family locally, and 50k-150k per family on Leonardo if runtime allows. No external fab data is required; the official generator is the intended extra data source.

For long Leonardo jobs, pass `--skip-if-complete` to reuse complete generated CSVs on retry. The Leonardo scripts also pass `--exact-count`, so reuse is accepted only when the existing CSV has exactly `COUNT_PER_FAMILY` sequences. Generation writes each CSV atomically in chunks, avoids duplicate sequences across chunks, and creates a sidecar `*.metadata.json` with count, seed, chunk size, duplicate count, max chunk attempts, and status. Reuse requires an existing successful generation metadata chain; a bare CSV without sidecar provenance is not accepted. If unique-sequence generation falls short, the partial temp CSV is removed and the metadata records the shortfall.

Audit corpus size before training:

```bash
python -m industrial_ai.audit_corpus --min-generated-per-family 50000
```

This writes `artifacts/corpus_audit/files.csv`, `families.csv`, and `summary.json`, including a corpus fingerprint, and exits non-zero when generated data is below the requested threshold. The Leonardo scripts also pass `--max-generated-per-family "${COUNT_PER_FAMILY}"`, so a stale oversized generated corpus cannot silently change the effective training corpus for a 50k or 150k run. When a generated-data threshold is requested, the audit also checks each generated CSV's `*.metadata.json` sidecar for a successful generation origin, matching family, and matching sequence count. `industrial_ai.prepare` reads generated CSVs when present, and the Leonardo generation/full-pipeline scripts run it after the generated-corpus audit so prepared stats match the training corpus.

Train a model:

```bash
python -m industrial_ai.train --model-size tiny --epochs 4 --batch-size 64 --device cuda
```

Supported model sizes: `tiny`, `small`, `medium`.

For retryable cluster runs, add `--skip-if-complete`. It reuses an existing checkpoint only when `model.pt`, `train_log.json`, and `train_summary.json` match the requested model size, hyperparameters, seed, corpus size, per-family sequence counts, corpus fingerprint, train-log hash, and, when source-bundle proof is active, the readiness source-bundle SHA. Leonardo GPU scripts also pass `--require-device`, so `--device cuda` fails instead of silently falling back to CPU if CUDA is unavailable.

Each training run writes:

- `checkpoints/<size>/model.pt`
- `checkpoints/<size>/train_log.json`
- `checkpoints/<size>/train_summary.json` with corpus, model, and train-log hashes
- `checkpoints/<size>/loss_curve.png` when `matplotlib` is installed

Use a trained checkpoint for next-step and completion candidate reranking:

```bash
python -m industrial_ai.infer \
  --valid-input data/eval/eval_input_valid.csv \
  --anomaly-input data/eval/eval_input_anomaly.csv \
  --completion-mode ensemble \
  --checkpoint checkpoints/small/model.pt \
  --transformer-device cuda \
  --out-dir submissions
```

Compare completion modes on a local dev set:

```bash
python -m industrial_ai.compare_completion
python -m industrial_ai.compare_rerankers
```

This writes per-mode completion predictions under `artifacts/completion_compare/` and baseline-vs-checkpoint reranker evidence under `artifacts/reranker_compare/`. Both commands emit `metrics.csv` and `metrics.json` with corpus size and fingerprint evidence; reranker comparison also emits `family_metrics.csv`, `REPORT.md`, and optional PNG plots when `matplotlib` is installed. Leonardo finalization runs completion comparison with `--require-checkpoint --require-transformer-available` and reranker comparison with `--selection-scope checkpoints --require-selected-checkpoint --require-checkpoints-available`, so baseline remains in the report but cannot be selected for the final package and missing or unloadable checkpoints fail before inference.

After `compare_rerankers` runs, `industrial_ai.infer --require-selected-checkpoint` uses the selected checkpoint from `artifacts/reranker_compare/metrics.json` when no explicit `--checkpoint` is provided. Exploratory/local inference can still fall back to no checkpoint scoring if baseline wins or if a checkpoint file exists but cannot be loaded by the transformer scorer. Strict Leonardo validation and packaging require `selection_scope=checkpoints` plus a selected checkpoint reranker, so the final package fails instead of silently accepting a baseline-selected run. Override inference with `--checkpoint checkpoints/<size>/model.pt` or `CHECKPOINT=... scripts/leonardo_infer.sh`. The Leonardo inference paths request a GPU, default `TRANSFORMER_DEVICE=cuda`, require a loadable checkpoint scorer, and fail before writing submissions if an explicit inference checkpoint path or live file SHA-256 does not match the selected reranker checkpoint in `artifacts/reranker_compare/metrics.json`.

Validate a completed Leonardo run:

```bash
python -m industrial_ai.validate_run --min-generated-per-family 50000 --require-submissions
```

This checks corpus audit thresholds, `tiny`/`small`/`medium` checkpoint summaries, exact checkpoint training counts and fingerprints against the audited corpus, checkpoint model hashes, checkpoint training device and minimum epoch count when required, exact reranker/completion comparison corpus sizes and fingerprints against the audit, exact inference corpus size and fingerprint against the audit, successful checkpoint loading in reranker comparison for every required checkpoint size, reranker/completion comparison example counts, required completion-comparison checkpoint size, selected checkpoint existence, selected reranker score ordering, checkpoint hash consistency, and non-empty submission CSVs. On Leonardo, `industrial_ai.checkpoint_audit` performs the checkpoint/corpus subset of these checks immediately after scaling training and before final comparisons, then the full validation repeats the checks before packaging. The full pipeline also requires the preflight JSON to prove raw data, generator loading, PyTorch, CUDA availability, CUDA-trained checkpoints, CUDA comparison/inference scorer use, checkpoint-reranker selection, and checkpoint epoch counts. It writes `artifacts/validation_summary.json` with the thresholds, derived run profile, audited family counts, pass/fail status, and any failures.
When submissions are required, it also checks `inference_summary.json`, verifies its row counts match the CSVs, and confirms final inference used the checkpoint selected by reranker comparison.

Leonardo scripts also write `artifacts/run_manifest.json` with the latest run stage and append every stage to `artifacts/run_manifest_events.jsonl`. These capture run parameters, the completion-comparison checkpoint (`checkpoints/medium/model.pt` for the full Leonardo run), derived run profile (`standard`, `max`, or `custom`), git state, Python/platform details, Slurm metadata when available, and key artifact paths, including shell-audit, readiness, checkpoint audit, validation, and package artifacts plus the package checksum sidecar. After readiness has run, each later manifest stage also embeds the verified source-bundle evidence from `leonardo_readiness.json`, including bundle SHA-256, manifest source, readiness pass state, and source-bundle requirement flag. Full, split-generation, and finalizer runs record `generation_prepared` after exact generated-corpus audit; full and finalizer runs append `checkpoint_audited` after checkpoint audit and `comparisons_complete` after medium completion comparison plus checkpoint-only reranker selection. Strict packaging/final audit require `generation_prepared`, `checkpoint_audited`, `comparisons_complete`, and the terminal packaged stage in order, and require post-readiness stages to carry the same source-bundle hash as packaged readiness when source-bundle proof is required. Validation summaries, checkpoint audits, package manifests, run manifests, and terminal manifest events all carry the same derived run profile, and strict packaging/final audit reject mismatches. In the full pipeline, the final successful stage is `packaged_with_submissions` when eval inputs are present, or `validated_without_submissions` when they are absent. Strict packaging can require that terminal manifest stage before writing the ZIP.
Preflight outputs are written under `artifacts/preflight*.json` and included in the submission package when available. When eval files are required, preflight records and validates their byte counts, required columns, headers, and row counts. When CUDA is required, preflight records CUDA availability, device count, device names, compute capability, memory, CUDA runtime version, and cuDNN version; strict final audit checks the packaged preflight evidence directly.

Package the final CSVs and evidence:

```bash
python -m industrial_ai.package_submission --include-evidence
python -m industrial_ai.verify_package
python -m industrial_ai.final_audit
python -m industrial_ai.verify_returned_package
python -m industrial_ai.run_evidence_report --count-profile standard
```

This writes `artifacts/submission_package/track1_submission.zip`, `track1_submission.zip.sha256`, and `package_manifest.json` with row counts and SHA-256 hashes for the packaged files. With `--include-evidence`, the package also includes Leonardo shell scripts, a Python source snapshot under `evidence/source_snapshot/`, shell-audit output, readiness output, eval staging manifest, source-bundle proof self-test output, preflight output, checkpoint audit output, run manifests, validation summary, corpus audit files, generated CSV metadata sidecars, reranker metrics, and checkpoint training summaries/logs. When readiness required source-bundle proof, strict packaging now fails before ZIP creation unless the ordered `generation_prepared`, `checkpoint_audited`, `comparisons_complete`, and terminal events carry the same bundle SHA as readiness, the source-bundle self-test passed, current Leonardo scripts and `industrial_ai/` source files match the per-file hashes in the readiness source-bundle manifest, and `package_manifest.json` records that source-bundle summary. When readiness and eval preflight evidence are required, strict packaging also requires `eval_staging_manifest.json` and compares staged eval hashes against readiness and preflight before writing the ZIP. Final audit repeats that package-only comparison after transfer. Final audit requires the script copies to be listed in the package manifest, recomputes their hashes, requires them to match the shell-audit hashes, and requires packaged Leonardo script bytes plus the packaged Python source snapshot to match the per-file source-bundle manifest hashes, so the returned package proves which bundled launch scripts and Python modules produced it. It also requires `reranker_compare/best_checkpoint.txt` to match the selected checkpoint recorded in `reranker_compare/metrics.json`. The full pipeline refreshes the package after the terminal packaged manifest stage so the ZIP evidence contains the final stage, then runs `industrial_ai.verify_package`, `industrial_ai.final_audit`, `industrial_ai.run_evidence_report`, and `industrial_ai.verify_returned_package` with the active `COUNT_PER_FAMILY`, checkpoint, transformer-device, and epoch thresholds before the job exits. After strict returned-package verification, the full-pipeline and finalizer scripts also write `artifacts/leonardo_return_packet.zip`, `.sha256`, and `leonardo_return_packet_manifest.json`. Prefer copying those three files back and verifying with `python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective`; the verifier checks the packet checksum, embedded/external manifests, required returned-run files, per-entry hashes, final objective flags, and the nested submission package. If copying only the package directory back from Leonardo, run `python -m industrial_ai.verify_package --package-dir artifacts/submission_package` again to verify the ZIP checksum, manifest, every archived entry, and any unpacked package/evidence files before upload.
For final Leonardo packaging, use `--require-evidence --required-min-generated-per-family 50000 --required-max-generated-per-family 50000 --required-min-reranker-count 240 --required-min-completion-compare-count 240 --required-min-train-epochs 6 --required-batch-size 96 --required-manifest-stage packaged_with_submissions --require-preflight-cuda --require-preflight-eval --required-transformer-device cuda --require-selected-checkpoint --require-generated-metadata --require-readiness --required-checkpoint-sizes tiny small medium` so missing shell audit, readiness, preflight, checkpoint audit, validation, reranker, completion comparison, corpus audit, generated metadata, training evidence, CUDA preflight validation, eval preflight validation, CUDA checkpoint validation, CUDA comparison/inference scorer validation, selected-checkpoint reranker validation, minimum epoch validation, batch-size validation, or terminal manifest evidence fails the job. Strict packaging also checks that validation and checkpoint audit covered the required generated-data minimum and maximum thresholds, derived run profile, reranker count, checkpoint sizes, training batch size, transformer scorer device, checkpoint-only reranker selection, selected-checkpoint reranker, and submissions, that every required checkpoint size has an available and eligible reranker row whose hash matches its checkpoint, that completion comparison used the expected checkpoint size, that the selected checkpoint's model size is covered by the required checkpoint evidence list, that the selected reranker was eligible and tied for the highest eligible selection score, and that live checkpoint file hashes still match the training, reranker, completion-comparison, and inference evidence at package time.
As a final pre-submit check on returned Leonardo artifacts, run `python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96`; it runs the ZIP verifier, `industrial_ai.final_audit`, and `industrial_ai.run_evidence_report`, then writes `artifacts/returned_package_verification.json` with the package, final-audit, and evidence-report hashes plus parsed checksum-sidecar and package-manifest source evidence, matching final-audit/report expected thresholds, `passed=true`, `objective_ready=true`, and `final_leonardo_objective_ready=true`. Use `--count-profile max` when verifying a 150k/family run, or pass matching `--min-generated-per-family` and `--max-generated-per-family` values for a custom exact target; the returned-package wrapper rejects ranged min/max bounds because Leonardo readiness and packaging evidence prove one launched `COUNT_PER_FAMILY`. With `--require-final-leonardo-objective`, the wrapper fails unless the evidence report proves the strict final Leonardo scope, so relaxed local smoke summaries cannot be accepted as final completion evidence. With `--require-source-bundle-proof`, the wrapper now passes that requirement into both final audit and the evidence report; this flag requires readiness evidence because the source-bundle hash is anchored there. The evidence report writes `artifacts/run_evidence_report.json` and `.md`, a compact objective summary covering generated counts, CUDA checkpoint training, packaged model/train-log hash evidence, checkpoint reranker selection, selected-reranker score ordering, completion/inference checkpoint use, submissions, package manifest, terminal run manifest, event-log stage order, readiness, source-bundle proof, generated launch-command proof, packaged script/source-snapshot hash proof, and final audit status; it prefers package evidence over same-named local artifacts by default so stale local files do not mask the returned archive. It can also be rerun directly with `python -m industrial_ai.run_evidence_report --count-profile standard --required-batch-size 96` or `--count-profile max --required-batch-size 96`. The final audit defaults to requiring CUDA and eval preflight evidence, CUDA checkpoint evidence, CUDA comparison/inference transformer-device evidence, selected-checkpoint reranker evidence, generated metadata sidecars, readiness evidence, and source-bundle proof, and checks the validation summary, derived run profile, terminal manifest stage, run-manifest event order, run-manifest source-bundle hash match, package-manifest source-bundle hash match, strict package manifest, packaged CUDA/eval preflight details, packaged corpus audit thresholds and family rows, packaged readiness thresholds, readiness eval-required mode, returned-package verification command, objective evidence-report command, resume guidance, dependency-safe split-job commands, readiness launch-command thresholds, readiness batch-size exports and verification flags, packaged script/source-snapshot hash proof, packaged checkpoint-audit/validation consistency, packaged checkpoint hashes against training summaries, packaged required checkpoint reranker rows and hashes, packaged completion-comparison checkpoint size, recorded validation/package completion-checkpoint-size requirements, packaged reranker pointer consistency, selected reranker score ordering, packaged execution evidence, validation/package threshold consistency, required evidence entries, required submission entries, local/package evidence agreement, and ZIP verifier together. Validation, manifest, event-log, corpus audit, and execution evidence can be read from the unpacked `evidence/` directory or directly from `track1_submission.zip`, so package-only verification works when the ZIP, checksum sidecar, and `package_manifest.json` are present either unpacked or as ZIP entries. For local smoke artifacts only, omit `--require-final-leonardo-objective` and pass `--required-transformer-device "" --no-require-selected-checkpoint --no-require-preflight-cuda --no-require-preflight-eval --no-require-generated-metadata --no-require-readiness --no-require-source-bundle-proof` with relaxed thresholds.

Completion modes:

- `prefix`: original exact-prefix/greedy baseline
- `retrieval`: longest-prefix and recent-context candidate retrieval
- `beam`: retrieval plus n-gram beam fallback
- `ensemble`: retrieval + beam + validator penalties + optional transformer checkpoint reranking

## Leonardo

Copy `.env.example` to `.env` locally and fill your Leonardo username/password outside git. Stage this repo on Leonardo, then use:

For the condensed launch, copy-back, and verification checklist, see [`LEONARDO_RUNBOOK.md`](LEONARDO_RUNBOOK.md).

Create a source bundle for upload to Leonardo:

```bash
python -m industrial_ai.leonardo_bundle
python -m industrial_ai.leonardo_bundle --include-eval
```

This writes `artifacts/leonardo_source_bundle.zip`, a `.sha256` sidecar, and `artifacts/leonardo_source_bundle_manifest.json`. The bundle includes source code, Slurm scripts, raw training data, placeholders, and docs, while excluding local runtime outputs, checkpoints, submissions, `.env`, and virtual environments. Use `--include-eval` only when the official eval CSVs are already staged locally and should be copied with the source bundle. The external manifest and the ZIP-embedded manifest include a `handoff` section listing the upload files, bundle/root verification commands, final readiness commands, deferred-eval pre-upload readiness commands, source-bundle self-test command, and handoff audit command. Root verification hashes every manifest-listed file and rejects unexpected source or launch-script files under `industrial_ai/` and `scripts/`; official eval CSVs can still be staged after unpacking under `data/eval/`, and runtime outputs can still be written under `artifacts/`. After upload, keep the ZIP, sidecar, and manifest together either at the staged repo root or under `artifacts/`; readiness, the handoff audit, and the Slurm scripts detect both layouts.
Before upload, run readiness with `--require-source-bundle` to prove the ZIP, checksum sidecar, manifest, and current working tree still match. After uploading and unpacking the bundle on Leonardo, verify the staged tree before submitting jobs:

```bash
python -m industrial_ai.leonardo_readiness --require-source-bundle
python -m industrial_ai.source_bundle_proof_selftest
python -m industrial_ai.leonardo_handoff --require-source-bundle
sha256sum -c leonardo_source_bundle.zip.sha256
unzip -o leonardo_source_bundle.zip
python -m industrial_ai.leonardo_bundle --verify-root . --manifest leonardo_source_bundle_manifest.json
```

The handoff audit also writes `artifacts/leonardo_handoff_checklist.md`, a bundle-specific checklist with the current bundle SHA, upload files, eval-staging command, readiness-generated launch commands, and returned-package verification commands. It also writes `artifacts/leonardo_transfer_packet.zip` with `.sha256` and manifest sidecars; that packet contains the source bundle, source-bundle sidecar, source-bundle manifest, handoff JSON, checklist, launch commands, readiness JSON, and source-bundle self-test result so the Leonardo upload can be moved as one file and verified before unpacking. After transfer, run `sha256sum -c leonardo_transfer_packet.zip.sha256`, unpack the transfer packet, verify and unpack `leonardo_source_bundle.zip`, then run `python -m industrial_ai.leonardo_handoff --verify-transfer-packet leonardo_transfer_packet.zip --verify-fresh-unpack` to recheck the packet manifest, required entries, per-entry hashes, and a temp fresh unpack of the nested source bundle before launching jobs.
On Leonardo, use the event-provided module stack or a virtual environment that provides CUDA-enabled PyTorch, then run `python -m industrial_ai.preflight --require-torch --require-cuda --out artifacts/preflight_leonardo_environment.json` before staging eval or submitting jobs.

You can also verify the uploaded ZIP before unpacking:

```bash
python -m industrial_ai.leonardo_bundle --verify-bundle leonardo_source_bundle.zip --sidecar leonardo_source_bundle.zip.sha256
```

```bash
sbatch scripts/leonardo_probe.sh
sbatch --export=ALL,COUNT_PER_FAMILY=50000,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh
sbatch --export=ALL,MODEL_SIZE=tiny,COUNT_PER_FAMILY=50000,EPOCHS=6,BATCH_SIZE=96,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train.sh
sbatch --export=ALL,COUNT_PER_FAMILY=50000,EPOCHS=6,BATCH_SIZE=96,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh
sbatch --export=ALL,VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_infer.sh
sbatch --export=ALL,COUNT_PER_FAMILY=50000,EPOCHS=6,RERANKER_VALID_PER_FAMILY=40,REQUIRE_EVAL=1,VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_finalize.sh
```

The repo tracks `artifacts/slurm/.gitkeep` so Slurm can open the configured `artifacts/slurm/*.log` output paths on a fresh checkout before the script body runs.
Before submitting, run `python -m industrial_ai.leonardo_readiness --require-eval --require-source-bundle` on the staged repo when official eval files are expected; it checks the launch scripts, tracked Slurm output directory, eval CSV schema, row counts, staging-manifest hashes, strict checkpoint-reranker gates, medium-checkpoint completion comparison, final-run thresholds, source-bundle freshness, and prints the exact probe, full-pipeline, split-job `sbatch`, dependency-safe split-job, resume guidance, and post-run verification commands. If the official eval CSVs are not available locally before upload, `--defer-eval-staging` can be used only for the pre-upload handoff command generation; after the official CSVs are staged into `data/eval/` on Leonardo with `industrial_ai.stage_eval_inputs`, rerun readiness without that flag before submitting jobs. Strict packaging, final audit, and evidence reporting reject final readiness evidence that still has `defer_eval_staging=true`. The readiness JSON also stores those commands under `commands`, `verification_commands`, and `resume_guidance`, plus `source_bundle` when bundle evidence is present; readiness also writes `artifacts/leonardo_launch_commands.sh`, which final packaging includes and final audit checks. Stale bundle evidence is a warning unless `--require-source-bundle` is used, and then generated full, split-generation, split-training, and finalizer commands export `REQUIRE_SOURCE_BUNDLE=1`, add explicit `SOURCE_BUNDLE` paths when the staged files are root-level rather than under `artifacts/`, the job readiness evidence requires the same proof, and recorded returned-package/evidence-report verification commands explicitly carry source-bundle proof and package-evidence precedence flags. Source-bundle readiness also records `python -m industrial_ai.source_bundle_proof_selftest`, and package/final/report checks reject readiness evidence that omits that tamper-detection self-test. Generated `--export` commands use `ALL,...` to preserve the submitted environment while setting run parameters, and `commands.split_jobs_with_dependencies` uses `sbatch --parsable` and `--dependency=afterok` so queued split jobs run only after the previous stage succeeds. The readiness helper supports `--count-profile standard` for the default 50k/family corpus and `--count-profile max` for the 150k/family upper target; direct `--count-per-family 50000` and `--count-per-family 150000` are normalized to those same profiles, while other direct overrides remain custom dry runs. Without `--require-eval`, the readiness command prints a no-submission full-pipeline command unless both eval files already exist.
For a local shell-script sanity check that does not require Bash, run `python -m industrial_ai.leonardo_shell_audit`; readiness runs the same audit and rejects missing shebangs, CRLF line endings, missing strict mode, missing launch guards, missing Slurm log directories, and common unbalanced quote or bracket mistakes. The audit records byte counts and SHA-256 hashes for each Leonardo script, and final audit requires the package to contain matching script copies under `evidence/scripts/`. Use repeated `--script path/to/file.sh` arguments to audit edited or temporary launch scripts directly.
Run `scripts/leonardo_probe.sh` first on the staged Leonardo checkout. It performs a small GPU smoke under `artifacts/probe/`: CUDA preflight, one `tiny` training epoch on CUDA, checkpoint completion/reranker comparisons, selected-checkpoint inference, and strict validation. That catches environment, CUDA, and checkpoint-loading problems before spending the larger generation/training allocation.
The Leonardo scripts source `scripts/leonardo_common.sh` and reject invalid launch settings early: final generation/training counts must be 50k-150k per family, final training must use at least 6 epochs, reranker comparisons must use at least 40 valid examples per family, and Leonardo inference must use CUDA transformer scoring.

For one queued job that generates a 50k-per-family corpus by default, trains `tiny`, `small`, and `medium`, compares rerankers, and runs official inference when eval files are present:

```bash
sbatch scripts/leonardo_probe.sh
sbatch --export=ALL,COUNT_PER_FAMILY=50000,EPOCHS=6,BATCH_SIZE=96,RERANKER_VALID_PER_FAMILY=40 scripts/leonardo_full_pipeline.sh
```

For the 150k/family upper target, first verify the staged repo and generated command:

```bash
python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle
sbatch scripts/leonardo_probe.sh
sbatch --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,RERANKER_VALID_PER_FAMILY=40,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1,VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_full_pipeline.sh
```

The standalone inference script also defaults to `data/eval/eval_input_valid.csv` and `data/eval/eval_input_anomaly.csv`, so launching it without explicit eval paths targets the official eval drop directory instead of local dev files. The full pipeline defaults `RERANKER_VALID_PER_FAMILY=40`, which creates 240 valid-task examples for reranker comparison and requires that count during final validation. If official eval inputs are expected, pass `REQUIRE_EVAL=1` or explicit `VALID_INPUT`/`ANOMALY_INPUT` paths; the full pipeline will then fail fast if either eval file is missing instead of falling back to a no-submission validation run.
For split jobs, run `scripts/leonardo_finalize.sh` after generation and `tiny`/`small`/`medium` training. The generation and training scripts rerun source-bundle readiness before producing artifacts when `REQUIRE_SOURCE_BUNDLE=1`, and the training jobs audit `COUNT_PER_FAMILY` before starting, so a stale, incomplete, or oversized generated corpus fails before GPU training. Finalization defaults to strict eval mode, and generated split-job commands include `REQUIRE_EVAL=1`; it reruns local dev comparisons, selects the checkpoint reranker, runs official inference, validates, packages, verifies, and runs final audit with the same strict gates as the full pipeline.
Safe resume rule: rerun the same full-pipeline or split generation/training command after an interruption. Generated CSVs are reused only when exact counts and metadata match, and checkpoints are reused only when training summaries match the requested model, hyperparameters, corpus counts, corpus fingerprint, and active source-bundle SHA. Use `generation_prepared`, `checkpoint_audited`, `comparisons_complete`, and the terminal package stage in `artifacts/run_manifest_events.jsonl` to see how far a run progressed. For split jobs, submit finalization only after the event log records `generation_prepared` plus `train_scaling_complete` or all three `train_tiny_complete`, `train_small_complete`, and `train_medium_complete` stages. When queueing split jobs together, use the readiness `split_jobs_with_dependencies` commands instead of submitting all commands independently.

Known hackathon defaults:

- partition: `boost_usr_prod`
- reservation: `s_tra_ncc`
- login host: `login01-ext.leonardo.cineca.it`

## Event Submission Checklist

Submit through the Tally form by Sunday 10:00:

- Team name
- Public GitHub repository URL: `https://github.com/j-avdeev/industrial-ai-process-logic`
- Slides PDF, max 10 slides: [`SLIDES.pdf`](SLIDES.pdf)
- Demo video or link, max 2 minutes: script and description in [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md)

Prepared submission support files:

- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) for final Tally checks
- [`SLIDES.md`](SLIDES.md) as editable slide source
- [`SLIDES.pdf`](SLIDES.pdf) as upload-ready slide deck
- [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) with a 2-minute recording plan and Tally description

Regenerate the slide PDF with:

```bash
python scripts/create_submission_slides_pdf.py
```

Repo requirements:

- public
- MIT licensed
- `README.md`
- `REPORT.md`
- dependency manifest
- no secrets
- clean-checkout runnable

Track-specific repo deliverables:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`
- packaged ZIP from `artifacts/submission_package/track1_submission.zip`
- training logs/checkpoints/loss curves or links
- run manifest from `artifacts/run_manifest.json` and `artifacts/run_manifest_events.jsonl`
- preflight output from `artifacts/preflight*.json`
- corpus audit artifacts from `artifacts/corpus_audit/`
- metrics report with per-family breakdown where possible
- baseline vs trained examples

## Data Attribution

Official challenge data and grammar are copied from:

https://github.com/Lumos-Data/zero_one_hack_01/tree/main/tracks/industrial-infineon/training_data
