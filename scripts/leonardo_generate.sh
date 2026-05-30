#!/usr/bin/env bash
#SBATCH --job-name=indai-generate
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/slurm/generate-%j.log

set -euo pipefail
mkdir -p artifacts/slurm data/generated

COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-10000}"

python -m industrial_ai.prepare
python -m industrial_ai.generate_extra --family mosfet --count "${COUNT_PER_FAMILY}" --seed 101 --output data/generated/MOSFET_extra.csv
python -m industrial_ai.generate_extra --family igbt --count "${COUNT_PER_FAMILY}" --seed 102 --output data/generated/IGBT_extra.csv
python -m industrial_ai.generate_extra --family ic --count "${COUNT_PER_FAMILY}" --seed 103 --output data/generated/IC_extra.csv
