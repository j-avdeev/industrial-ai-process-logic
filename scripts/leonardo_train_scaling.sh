#!/usr/bin/env bash
#SBATCH --job-name=indai-scale
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/slurm/train-scaling-%j.log

set -euo pipefail
mkdir -p artifacts/slurm checkpoints

EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-96}"

for MODEL_SIZE in tiny small medium; do
  echo "Training ${MODEL_SIZE}"
  python -m industrial_ai.train \
    --model-size "${MODEL_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda
done

