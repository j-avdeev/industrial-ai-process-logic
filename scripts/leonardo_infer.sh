#!/usr/bin/env bash
#SBATCH --job-name=indai-infer
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/slurm/infer-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm submissions

VALID_INPUT="${VALID_INPUT:-data/eval/eval_input_valid.csv}"
ANOMALY_INPUT="${ANOMALY_INPUT:-data/eval/eval_input_anomaly.csv}"
COMPLETION_MODE="${COMPLETION_MODE:-ensemble}"
TRANSFORMER_DEVICE="${TRANSFORMER_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
require_choice COMPLETION_MODE "${COMPLETION_MODE}" auto prefix retrieval beam ensemble
require_choice TRANSFORMER_DEVICE "${TRANSFORMER_DEVICE}" cuda

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

python -m industrial_ai.run_manifest \
  --stage infer_start \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "COMPLETION_MODE=${COMPLETION_MODE}" \
  --set "TRANSFORMER_DEVICE=${TRANSFORMER_DEVICE}" \
  --set "CHECKPOINT=${CHECKPOINT}"
python -m industrial_ai.preflight \
  --require-torch \
  --require-cuda \
  --valid-input "${VALID_INPUT}" \
  --anomaly-input "${ANOMALY_INPUT}" \
  --require-eval \
  --out artifacts/preflight_infer.json

python -m industrial_ai.infer \
  --valid-input "${VALID_INPUT}" \
  --anomaly-input "${ANOMALY_INPUT}" \
  --completion-mode "${COMPLETION_MODE}" \
  "${CHECKPOINT_ARGS[@]}" \
  --transformer-device "${TRANSFORMER_DEVICE}" \
  --require-checkpoint \
  --require-transformer-available \
  --require-selected-checkpoint \
  --out-dir submissions
python -m industrial_ai.run_manifest \
  --stage infer_complete \
  --set "VALID_INPUT=${VALID_INPUT}" \
  --set "ANOMALY_INPUT=${ANOMALY_INPUT}" \
  --set "COMPLETION_MODE=${COMPLETION_MODE}" \
  --set "TRANSFORMER_DEVICE=${TRANSFORMER_DEVICE}" \
  --set "CHECKPOINT=${CHECKPOINT}"
