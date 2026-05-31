# Leonardo Runbook

This runbook is for the final Track 1 improvement path: generate a large official-generator corpus on Leonardo, train `tiny`, `small`, and `medium` on CUDA, select a checkpoint reranker, package submissions, and verify the returned package.
The final launch scripts reject `COUNT_PER_FAMILY` outside the 50k-150k objective range.

## Preconditions

- Official eval files are staged at `data/eval/eval_input_valid.csv` and `data/eval/eval_input_anomaly.csv`.
- `artifacts/slurm/` exists in the checkout.
- Source bundle is built locally when copying the repo manually:

```bash
python -m industrial_ai.leonardo_bundle
```

This writes `artifacts/leonardo_source_bundle.zip`, `artifacts/leonardo_source_bundle.zip.sha256`, and `artifacts/leonardo_source_bundle_manifest.json`. Upload the ZIP, sidecar, and manifest to Leonardo, verify the sidecar hash, unpack it into the run directory, then stage official eval CSVs under `data/eval/` unless the bundle was created with `--include-eval`. The external and ZIP-embedded manifests include a `handoff` section with the upload files, verify commands, final source-bundle readiness commands, and deferred-eval pre-upload readiness commands. Root verification hashes every manifest-listed file and rejects unexpected source or launch-script files under `industrial_ai/` and `scripts/`; official eval CSVs under `data/eval/` and runtime outputs under `artifacts/` remain allowed after unpacking. The Slurm scripts and readiness check accept the bundle files either at the repo root or under `artifacts/`; keep all three files together.

Before upload, verify the current tree still matches the bundle:

```bash
python -m industrial_ai.leonardo_readiness --require-source-bundle
python -m industrial_ai.source_bundle_proof_selftest
python -m industrial_ai.leonardo_handoff --require-source-bundle
```

The handoff audit writes `artifacts/leonardo_handoff.json`, `artifacts/leonardo_handoff_checklist.md`, and `artifacts/leonardo_transfer_packet.zip` with `.sha256` and manifest sidecars. Use the checklist as the current bundle-specific upload, eval-staging, launch, and returned-package verification guide; it records the exact bundle SHA and the readiness-generated commands. The transfer packet is a convenience wrapper containing the source bundle, source-bundle sidecar, source-bundle manifest, handoff JSON, checklist, launch commands, readiness JSON, and source-bundle self-test result.

Before submitting jobs on Leonardo, verify the uploaded ZIP and unpacked source tree:

```bash
sha256sum -c leonardo_source_bundle.zip.sha256
unzip -o leonardo_source_bundle.zip
python -m industrial_ai.leonardo_bundle --verify-bundle leonardo_source_bundle.zip --sidecar leonardo_source_bundle.zip.sha256
python -m industrial_ai.leonardo_bundle --verify-root . --manifest leonardo_source_bundle_manifest.json
```

If using the single transfer packet instead, verify and unpack it before running the source-bundle commands:

```bash
sha256sum -c leonardo_transfer_packet.zip.sha256
unzip -o leonardo_transfer_packet.zip
sha256sum -c leonardo_source_bundle.zip.sha256
unzip -o leonardo_source_bundle.zip
python -m industrial_ai.leonardo_handoff --verify-transfer-packet leonardo_transfer_packet.zip --verify-fresh-unpack
```

Prepare the Python environment with the event-provided module stack or a virtual environment that provides CUDA-enabled PyTorch, then run the cheap environment preflight:

```bash
python --version
python -m pip install -r requirements.txt
python -m industrial_ai.preflight --require-torch --require-cuda --out artifacts/preflight_leonardo_environment.json
```

If the official eval CSVs are not available locally before upload, generate the pre-upload command/handoff evidence with:

```bash
python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --defer-eval-staging --require-source-bundle
python -m industrial_ai.leonardo_handoff --require-source-bundle
```

That mode is only for the upload handoff. After the official CSVs are available on Leonardo, stage them with the checked copy helper, then rerun readiness without `--defer-eval-staging`; packaging, final audit, and the evidence report reject final readiness evidence that still has deferred eval staging enabled.

```bash
python -m industrial_ai.stage_eval_inputs \
  --valid-input /path/to/official/eval_input_valid.csv \
  --anomaly-input /path/to/official/eval_input_anomaly.csv

python -m industrial_ai.preflight \
  --valid-input data/eval/eval_input_valid.csv \
  --anomaly-input data/eval/eval_input_anomaly.csv \
  --require-eval \
  --out artifacts/preflight_eval_staged.json

python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle
```

The staging command writes `artifacts/eval_staging_manifest.json` with row counts and SHA-256 hashes for the source and staged files. Job-time readiness without `--defer-eval-staging` requires that manifest to match the current `data/eval/` file hashes, so a stale hand copy fails before the expensive job starts. If the source bundle was created with `--include-eval`, still keep or regenerate `artifacts/eval_staging_manifest.json` after unpacking; the bundle includes the manifest only when it exists at bundle creation time.
- The staged repo passes:

```bash
python -m industrial_ai.leonardo_shell_audit
python -m industrial_ai.leonardo_readiness --count-profile standard --require-eval --require-source-bundle
```

For the upper target, use:

```bash
python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle
```

## Standard 50k Run

Submit the probe first. Only submit the full run after the probe succeeds.

```bash
sbatch scripts/leonardo_probe.sh
sbatch --export=ALL,COUNT_PER_FAMILY=50000,EPOCHS=6,BATCH_SIZE=96,RERANKER_VALID_PER_FAMILY=40,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1,VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_full_pipeline.sh
```

## Max 150k Run

```bash
sbatch scripts/leonardo_probe.sh
sbatch --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,RERANKER_VALID_PER_FAMILY=40,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1,VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_full_pipeline.sh
```

## Split-Job Alternative

Use the dependency-safe commands printed by:

```bash
python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --require-source-bundle
```

They queue probe, generation, training, and finalization with `afterok` dependencies so each stage runs only after the previous stage succeeds.
The same readiness run writes `artifacts/leonardo_launch_commands.sh`; use that generated script as the authoritative command checklist when copying commands into the scheduler. With `--require-source-bundle`, generated full, split-generation, split-training, and finalizer commands export `REQUIRE_SOURCE_BUNDLE=1` so each split stage reruns source-bundle readiness before producing artifacts and the readiness evidence packaged after the job proves the uploaded source bundle still matches the run tree. If readiness finds root-level bundle files, it also writes explicit `SOURCE_BUNDLE` and `SOURCE_BUNDLE_MANIFEST` exports into those commands. The generated returned-package and evidence-report verification commands include the source-bundle proof and package-evidence precedence flags expected by final audit.

## Expected Terminal Evidence

After a successful run, these artifacts must exist in `artifacts/`:

- `run_manifest.json` with stage `packaged_with_submissions`
- `run_manifest_events.jsonl` containing `generation_prepared`, `checkpoint_audited`, `comparisons_complete`, and `packaged_with_submissions` in that order
- source-bundle SHA evidence in those post-readiness manifest stages matching packaged `leonardo_readiness.json` when `--require-source-bundle` was used
- packaged Leonardo script bytes and `evidence/source_snapshot/industrial_ai/` source bytes matching the per-file hashes from the readiness source-bundle manifest, so script or Python edits after readiness are rejected
- `validation_summary.json` with `passed=true`, `require_submissions=true`, `require_preflight_eval=true`, and matching `run_profile`
- `checkpoint_audit.json` with `passed=true` and matching `run_profile`
- generated metadata sidecars whose `output_sha256` values match the audited generated CSV file hashes
- `tiny`/`small`/`medium` train summaries with model and train-log SHA-256 hashes matching the packaged files
- `submission_package/track1_submission.zip`
- `submission_package/track1_submission.zip.sha256`
- `submission_package/package_manifest.json` with the source-bundle SHA summary when `--require-source-bundle` was used
- `run_evidence_report.json` with `objective_ready=true`, `objective_scope=final_leonardo`, and `final_leonardo_objective_ready=true`
- `returned_package_verification.json` with `passed=true`, `objective_ready=true`, and `final_leonardo_objective_ready=true`; the full-pipeline and finalizer scripts write it as their terminal verification step
- `leonardo_return_packet.zip`, `.sha256`, and `leonardo_return_packet_manifest.json`; these wrap the verified returned package and summary evidence for copy-back
- `leonardo_launch_commands.sh` with generated launch and post-run verification commands
- `source_bundle_proof_selftest.json` with `passed=true`

The package evidence must include preflight, readiness, eval staging manifest, source-bundle proof self-test, launch commands, shell audit, run manifests, validation summary, checkpoint audit, corpus audit, reranker/completion metrics, generated metadata, and `tiny`/`small`/`medium` train summaries/logs. Strict packaging compares eval staging hashes against readiness and preflight before writing the ZIP, and final audit repeats the check after transfer.

## Copy Back And Verify

Copy back the whole `artifacts/submission_package/` directory. At minimum, the returned directory must contain:

- `track1_submission.zip`
- `track1_submission.zip.sha256`
- `package_manifest.json`

Verify the returned package locally:

```bash
python -m industrial_ai.verify_package --package-dir artifacts/submission_package
python -m industrial_ai.verify_returned_package --count-profile standard --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96
```

The returned-package verifier writes `artifacts/returned_package_verification.json`, tying the ZIP, checksum sidecar, package manifest source, final audit, evidence report hashes, and expected verification thresholds to a final `passed=true` status.
For a custom Leonardo count, pass the exact launched count as both `--min-generated-per-family` and `--max-generated-per-family`; ranged min/max bounds are rejected because readiness and package evidence prove a single `COUNT_PER_FAMILY`.
If copying back the single return packet instead, copy `artifacts/leonardo_return_packet.zip`, `artifacts/leonardo_return_packet.zip.sha256`, and `artifacts/leonardo_return_packet_manifest.json`, then run:

```bash
python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective
```

For the 150k run:

```bash
python -m industrial_ai.verify_package --package-dir artifacts/submission_package
python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96
```

Package-only verification is supported when the unpacked `evidence/` directory is absent; the verifier reads evidence directly from `track1_submission.zip`.
The returned-package verifier writes `artifacts/final_audit_summary.json`, `artifacts/run_evidence_report.json`, and `.md`; the evidence report is a compact objective summary, including explicit readiness, source-bundle proof, packaged checkpoint model/train-log hash checks, generated launch-command checks, and packaged script/source-snapshot hash checks.
When `--require-source-bundle` is used, readiness also records `python -m industrial_ai.source_bundle_proof_selftest` in the generated post-run verification commands, and strict packaging/final audit expect that command in `leonardo_launch_commands.sh`. Readiness-generated verification commands include `--required-batch-size 96` and `--require-final-leonardo-objective` for the default run, so returned-package verification proves the checkpoints were trained with the same batch-size setting used by the submitted jobs and rejects relaxed smoke evidence as final completion evidence.

## Completion Criteria

The goal is complete only when the returned package verification and run evidence report pass for a real Leonardo run at the intended profile, `final_leonardo_objective_ready=true`, and the package contains submissions produced after CUDA training for all requested checkpoints. A local tiny smoke run with completion exact match `0.0000` is only a pipeline check, not the final improvement run.
