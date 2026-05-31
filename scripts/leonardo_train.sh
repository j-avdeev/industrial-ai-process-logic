#!/usr/bin/env bash
#SBATCH --job-name=indai-train
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/slurm/train-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm checkpoints

MODEL_SIZE="${MODEL_SIZE:-tiny}"
COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-50000}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-96}"
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
require_choice MODEL_SIZE "${MODEL_SIZE}" tiny small medium
require_min_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 50000
require_max_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 150000
require_min_int EPOCHS "${EPOCHS}" 6
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_choice REQUIRE_SOURCE_BUNDLE "${REQUIRE_SOURCE_BUNDLE}" 0 1

READINESS_ARGS=(
  --count-per-family "${COUNT_PER_FAMILY}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --source-bundle "${SOURCE_BUNDLE}"
  --source-bundle-manifest "${SOURCE_BUNDLE_MANIFEST}"
  --out artifacts/leonardo_readiness.json
)
if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
  READINESS_ARGS+=(--require-source-bundle)
fi

python -m industrial_ai.leonardo_readiness "${READINESS_ARGS[@]}"
python -m industrial_ai.run_manifest \
  --stage "train_${MODEL_SIZE}_start" \
  --set "MODEL_SIZE=${MODEL_SIZE}" \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.preflight --require-torch --require-cuda --out "artifacts/preflight_train_${MODEL_SIZE}.json"
python -m industrial_ai.audit_corpus \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}"

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
python -m industrial_ai.run_manifest \
  --stage "train_${MODEL_SIZE}_complete" \
  --set "MODEL_SIZE=${MODEL_SIZE}" \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "EPOCHS=${EPOCHS}" \
  --set "BATCH_SIZE=${BATCH_SIZE}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
