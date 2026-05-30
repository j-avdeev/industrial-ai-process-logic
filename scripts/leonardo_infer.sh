#!/usr/bin/env bash
#SBATCH --job-name=indai-infer
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/slurm/infer-%j.log

set -euo pipefail
mkdir -p artifacts/slurm submissions

VALID_INPUT="${VALID_INPUT:-data/dev/eval_input_valid.csv}"
ANOMALY_INPUT="${ANOMALY_INPUT:-data/dev/eval_input_anomaly.csv}"

python -m industrial_ai.infer --valid-input "${VALID_INPUT}" --anomaly-input "${ANOMALY_INPUT}" --out-dir submissions

