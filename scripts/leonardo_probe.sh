#!/usr/bin/env bash
#SBATCH --job-name=indai-probe
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/slurm/probe-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm artifacts/probe

echo "Host: $(hostname)"
echo "User: ${USER}"
echo "Scratch: ${SCRATCH:-unset}"
echo "Public: ${PUBLIC:-unset}"
python --version
which python
module list || true
sinfo -p "${HPC_SLURM_PARTITION:-boost_usr_prod}" || true

python -m industrial_ai.run_manifest \
  --out artifacts/probe/run_manifest.json \
  --events-out artifacts/probe/run_manifest_events.jsonl \
  --artifacts-dir artifacts/probe \
  --checkpoint-dir artifacts/probe/checkpoints \
  --submission-dir artifacts/probe/submissions \
  --stage probe_start
python -m industrial_ai.leonardo_shell_audit --out artifacts/probe/leonardo_shell_audit.json
python -m industrial_ai.preflight \
  --require-torch \
  --require-cuda \
  --out artifacts/probe/preflight_full_pipeline.json
python -m industrial_ai.prepare
python -m industrial_ai.audit_corpus \
  --min-generated-per-family 0 \
  --out-dir artifacts/probe/corpus_audit
python -m industrial_ai.make_devset \
  --out-dir artifacts/probe/dev \
  --valid-per-family 1 \
  --anomaly-valid-per-family 1 \
  --anomaly-invalid-per-family 1
python -m industrial_ai.train \
  --model-size tiny \
  --epochs 1 \
  --batch-size 1024 \
  --device cuda \
  --require-device \
  --out-dir artifacts/probe/checkpoints \
  --skip-if-complete
python -m industrial_ai.compare_completion \
  --dev-dir artifacts/probe/dev \
  --out-dir artifacts/probe/completion_compare \
  --checkpoint artifacts/probe/checkpoints/tiny/model.pt \
  --transformer-device cuda \
  --require-checkpoint \
  --require-transformer-available \
  --max-examples 6
python -m industrial_ai.compare_rerankers \
  --dev-dir artifacts/probe/dev \
  --out-dir artifacts/probe/reranker_compare \
  --checkpoints artifacts/probe/checkpoints/tiny/model.pt \
  --transformer-device cuda \
  --selection-scope checkpoints \
  --require-selected-checkpoint \
  --require-checkpoints-available \
  --max-examples 6
python -m industrial_ai.infer \
  --valid-input artifacts/probe/dev/eval_input_valid.csv \
  --anomaly-input artifacts/probe/dev/eval_input_anomaly.csv \
  --out-dir artifacts/probe/submissions \
  --checkpoint artifacts/probe/checkpoints/tiny/model.pt \
  --reranker-metrics artifacts/probe/reranker_compare/metrics.json \
  --completion-mode ensemble \
  --transformer-device cuda \
  --require-checkpoint \
  --require-transformer-available \
  --require-selected-checkpoint
python -m industrial_ai.validate_run \
  --artifacts-dir artifacts/probe \
  --checkpoint-dir artifacts/probe/checkpoints \
  --submission-dir artifacts/probe/submissions \
  --model-sizes tiny \
  --min-generated-per-family 0 \
  --min-reranker-count 6 \
  --min-completion-compare-count 6 \
  --require-preflight \
  --require-preflight-torch \
  --require-preflight-cuda \
  --require-checkpoint-device cuda \
  --require-transformer-device cuda \
  --require-selected-checkpoint \
  --min-train-epochs 1 \
  --require-submissions
python -m industrial_ai.run_manifest \
  --out artifacts/probe/run_manifest.json \
  --events-out artifacts/probe/run_manifest_events.jsonl \
  --artifacts-dir artifacts/probe \
  --checkpoint-dir artifacts/probe/checkpoints \
  --submission-dir artifacts/probe/submissions \
  --stage probe_validated
