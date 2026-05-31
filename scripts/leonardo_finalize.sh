#!/usr/bin/env bash
#SBATCH --job-name=indai-final
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/slurm/finalize-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm submissions

COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-50000}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-96}"
RERANKER_VALID_PER_FAMILY="${RERANKER_VALID_PER_FAMILY:-40}"
VALID_INPUT="${VALID_INPUT:-data/eval/eval_input_valid.csv}"
ANOMALY_INPUT="${ANOMALY_INPUT:-data/eval/eval_input_anomaly.csv}"
REQUIRE_EVAL="${REQUIRE_EVAL:-1}"
REQUIRE_SOURCE_BUNDLE="${REQUIRE_SOURCE_BUNDLE:-0}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-}"
SOURCE_BUNDLE_MANIFEST="${SOURCE_BUNDLE_MANIFEST:-}"
if [[ -z "${SOURCE_BUNDLE}" ]]; then
  if [[ -f artifacts/leonardo_source_bundle.zip ]]; then
    SOURCE_BUNDLE="artifacts/leonardo_source_bundle.zip"
  else
    SOURCE_BUNDLE="leonardo_source_bundle.zip"
  fi
fi
if [[ -z "${SOURCE_BUNDLE_MANIFEST}" ]]; then
  if [[ -f artifacts/leonardo_source_bundle_manifest.json ]]; then
    SOURCE_BUNDLE_MANIFEST="artifacts/leonardo_source_bundle_manifest.json"
  else
    SOURCE_BUNDLE_MANIFEST="leonardo_source_bundle_manifest.json"
  fi
fi
require_min_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 50000
require_max_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 150000
require_min_int EPOCHS "${EPOCHS}" 6
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_min_int RERANKER_VALID_PER_FAMILY "${RERANKER_VALID_PER_FAMILY}" 40
require_choice REQUIRE_EVAL "${REQUIRE_EVAL}" 0 1
require_choice REQUIRE_SOURCE_BUNDLE "${REQUIRE_SOURCE_BUNDLE}" 0 1
SOURCE_BUNDLE_PROOF_ARGS=(--no-require-source-bundle-proof)
if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
  SOURCE_BUNDLE_PROOF_ARGS=(--require-source-bundle-proof)
fi
READINESS_ARGS=(
  --count-per-family "${COUNT_PER_FAMILY}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --reranker-valid-per-family "${RERANKER_VALID_PER_FAMILY}"
  --valid-input "${VALID_INPUT}"
  --anomaly-input "${ANOMALY_INPUT}"
  --source-bundle "${SOURCE_BUNDLE}"
  --source-bundle-manifest "${SOURCE_BUNDLE_MANIFEST}"
  --require-eval
  --out artifacts/leonardo_readiness.json
)
if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
  READINESS_ARGS+=(--require-source-bundle)
fi

python -m industrial_ai.run_manifest \
  --stage finalize_start \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.leonardo_shell_audit --out artifacts/leonardo_shell_audit.json
python -m industrial_ai.leonardo_readiness "${READINESS_ARGS[@]}"
python -m industrial_ai.source_bundle_proof_selftest --out artifacts/source_bundle_proof_selftest.json
python -m industrial_ai.preflight \
  --require-torch \
  --require-cuda \
  --require-eval \
  --valid-input "${VALID_INPUT}" \
  --anomaly-input "${ANOMALY_INPUT}" \
  --out artifacts/preflight_full_pipeline.json

python -m industrial_ai.audit_corpus \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}"
python -m industrial_ai.run_manifest \
  --stage generation_prepared \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.checkpoint_audit \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}" \
  --require-checkpoint-device cuda \
  --min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}"
python -m industrial_ai.run_manifest \
  --stage checkpoint_audited \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.make_devset \
  --valid-per-family "${RERANKER_VALID_PER_FAMILY}" \
  --anomaly-valid-per-family "${RERANKER_VALID_PER_FAMILY}" \
  --anomaly-invalid-per-family "${RERANKER_VALID_PER_FAMILY}"
python -m industrial_ai.compare_completion \
  --checkpoint checkpoints/medium/model.pt \
  --transformer-device cuda \
  --require-checkpoint \
  --require-transformer-available
python -m industrial_ai.compare_rerankers \
  --transformer-device cuda \
  --selection-scope checkpoints \
  --require-selected-checkpoint \
  --require-checkpoints-available
python -m industrial_ai.run_manifest \
  --stage comparisons_complete \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.infer \
  --valid-input "${VALID_INPUT}" \
  --anomaly-input "${ANOMALY_INPUT}" \
  --completion-mode ensemble \
  --transformer-device cuda \
  --require-checkpoint \
  --require-transformer-available \
  --require-selected-checkpoint \
  --out-dir submissions

python -m industrial_ai.validate_run \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}" \
  --min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --require-preflight \
  --require-preflight-torch \
  --require-preflight-cuda \
  --require-preflight-eval \
  --require-checkpoint-device cuda \
  --require-transformer-device cuda \
  --require-selected-checkpoint \
  --min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --require-submissions
python -m industrial_ai.run_manifest \
  --stage validated_with_submissions \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"

python -m industrial_ai.package_submission \
  --checkpoint-dir checkpoints \
  --generated-dir data/generated \
  --include-evidence \
  --require-evidence \
  --required-min-generated-per-family "${COUNT_PER_FAMILY}" \
  --required-max-generated-per-family "${COUNT_PER_FAMILY}" \
  --required-min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --required-min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --required-min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --required-transformer-device cuda \
  --require-selected-checkpoint \
  --require-preflight-cuda \
  --require-preflight-eval \
  --require-generated-metadata \
  --require-readiness \
  --required-checkpoint-sizes tiny small medium
python -m industrial_ai.run_manifest \
  --stage packaged_with_submissions \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"

python -m industrial_ai.package_submission \
  --checkpoint-dir checkpoints \
  --generated-dir data/generated \
  --include-evidence \
  --require-evidence \
  --required-min-generated-per-family "${COUNT_PER_FAMILY}" \
  --required-max-generated-per-family "${COUNT_PER_FAMILY}" \
  --required-min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --required-min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --required-min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --required-manifest-stage packaged_with_submissions \
  --required-transformer-device cuda \
  --require-selected-checkpoint \
  --require-preflight-cuda \
  --require-preflight-eval \
  --require-generated-metadata \
  --require-readiness \
  --required-checkpoint-sizes tiny small medium
python -m industrial_ai.verify_package --package-dir artifacts/submission_package
python -m industrial_ai.final_audit \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}" \
  --min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --required-transformer-device cuda \
  --require-selected-checkpoint \
  --require-preflight-cuda \
  --require-preflight-eval \
  --require-generated-metadata \
  --require-readiness \
  "${SOURCE_BUNDLE_PROOF_ARGS[@]}" \
  --required-checkpoint-sizes tiny small medium
python -m industrial_ai.run_evidence_report \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}" \
  --min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --required-checkpoint-sizes tiny small medium \
  --required-transformer-device cuda \
  --required-manifest-stage packaged_with_submissions \
  --require-readiness \
  "${SOURCE_BUNDLE_PROOF_ARGS[@]}" \
  --prefer-package-evidence
python -m industrial_ai.verify_returned_package \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}" \
  --min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
  --min-train-epochs "${EPOCHS}" \
  --required-batch-size "${BATCH_SIZE}" \
  --required-checkpoint-sizes tiny small medium \
  --required-transformer-device cuda \
  --required-manifest-stage packaged_with_submissions \
  --require-selected-checkpoint \
  --require-preflight-cuda \
  --require-preflight-eval \
  --require-generated-metadata \
  --require-readiness \
  --require-final-leonardo-objective \
  "${SOURCE_BUNDLE_PROOF_ARGS[@]}"
python -m industrial_ai.leonardo_return_packet --require-final-leonardo-objective
