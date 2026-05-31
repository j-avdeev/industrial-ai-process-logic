#!/usr/bin/env bash
#SBATCH --job-name=indai-full
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/slurm/full-pipeline-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm data/generated checkpoints submissions

COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-50000}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-96}"
RERANKER_VALID_PER_FAMILY="${RERANKER_VALID_PER_FAMILY:-40}"
VALID_INPUT_WAS_SET="${VALID_INPUT+x}"
ANOMALY_INPUT_WAS_SET="${ANOMALY_INPUT+x}"
VALID_INPUT="${VALID_INPUT:-data/eval/eval_input_valid.csv}"
ANOMALY_INPUT="${ANOMALY_INPUT:-data/eval/eval_input_anomaly.csv}"
REQUIRE_EVAL="${REQUIRE_EVAL:-0}"
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

EVAL_PREFLIGHT_ARGS=()
if [[ "${REQUIRE_EVAL}" == "1" || -n "${VALID_INPUT_WAS_SET}" || -n "${ANOMALY_INPUT_WAS_SET}" ]]; then
  EVAL_PREFLIGHT_ARGS=(--require-eval)
elif [[ -f "${VALID_INPUT}" || -f "${ANOMALY_INPUT}" ]]; then
  EVAL_PREFLIGHT_ARGS=(--require-eval)
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
  --out artifacts/leonardo_readiness.json
)
if (( ${#EVAL_PREFLIGHT_ARGS[@]} > 0 )); then
  READINESS_ARGS+=(--require-eval)
fi
if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
  READINESS_ARGS+=(--require-source-bundle)
fi

python -m industrial_ai.run_manifest \
  --stage start \
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
  --valid-input "${VALID_INPUT}" \
  --anomaly-input "${ANOMALY_INPUT}" \
  "${EVAL_PREFLIGHT_ARGS[@]}" \
  --out artifacts/preflight_full_pipeline.json

python -m industrial_ai.generate_extra --family mosfet --count "${COUNT_PER_FAMILY}" --seed 101 --output data/generated/MOSFET_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.generate_extra --family igbt --count "${COUNT_PER_FAMILY}" --seed 102 --output data/generated/IGBT_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.generate_extra --family ic --count "${COUNT_PER_FAMILY}" --seed 103 --output data/generated/IC_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.audit_corpus \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}"
python -m industrial_ai.prepare
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

for MODEL_SIZE in tiny small medium; do
  echo "Training ${MODEL_SIZE}"
  TRAIN_ARGS=(
    --model-size "${MODEL_SIZE}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --device cuda
    --require-device
    --skip-if-complete
  )
  if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
    TRAIN_ARGS+=(--require-source-bundle-proof)
  fi
  python -m industrial_ai.train "${TRAIN_ARGS[@]}"
done

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
python -m industrial_ai.run_manifest \
  --stage after_training \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
  --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"

if [[ -f "${VALID_INPUT}" && -f "${ANOMALY_INPUT}" ]]; then
  python -m industrial_ai.infer \
    --valid-input "${VALID_INPUT}" \
    --anomaly-input "${ANOMALY_INPUT}" \
    --completion-mode ensemble \
    --transformer-device cuda \
    --require-checkpoint \
    --require-transformer-available \
    --require-selected-checkpoint \
    --out-dir submissions
  python -m industrial_ai.run_manifest \
    --stage complete_with_submissions \
    --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
    --set "EPOCHS=${EPOCHS}" \
    --set "BATCH_SIZE=${BATCH_SIZE}" \
    --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
    --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
    --set "VALID_INPUT=${VALID_INPUT}" \
    --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
    --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
    --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
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
else
  echo "Skipping official eval inference; missing ${VALID_INPUT} or ${ANOMALY_INPUT}"
  python -m industrial_ai.run_manifest \
    --stage complete_without_submissions \
    --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
    --set "EPOCHS=${EPOCHS}" \
    --set "BATCH_SIZE=${BATCH_SIZE}" \
    --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
    --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
    --set "VALID_INPUT=${VALID_INPUT}" \
    --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
    --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
    --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
  python -m industrial_ai.validate_run \
    --min-generated-per-family "${COUNT_PER_FAMILY}" \
    --max-generated-per-family "${COUNT_PER_FAMILY}" \
    --min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
    --min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))" \
    --require-preflight \
    --require-preflight-torch \
    --require-preflight-cuda \
    --require-checkpoint-device cuda \
    --require-transformer-device cuda \
    --require-selected-checkpoint \
    --min-train-epochs "${EPOCHS}" \
    --required-batch-size "${BATCH_SIZE}"
  python -m industrial_ai.run_manifest \
    --stage validated_without_submissions \
    --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
    --set "EPOCHS=${EPOCHS}" \
    --set "BATCH_SIZE=${BATCH_SIZE}" \
    --set "RERANKER_VALID_PER_FAMILY=${RERANKER_VALID_PER_FAMILY}" \
    --set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt" \
    --set "VALID_INPUT=${VALID_INPUT}" \
    --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
    --set "REQUIRE_EVAL=${REQUIRE_EVAL}" \
    --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
fi
