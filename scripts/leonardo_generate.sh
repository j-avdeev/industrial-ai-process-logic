#!/usr/bin/env bash
#SBATCH --job-name=indai-generate
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/slurm/generate-%j.log

set -euo pipefail
mkdir -p artifacts/slurm data/generated

python -m industrial_ai.prepare
python -m industrial_ai.generate_extra --family mosfet --count 10000 --seed 101 --output data/generated/MOSFET_extra.csv
python -m industrial_ai.generate_extra --family igbt --count 10000 --seed 102 --output data/generated/IGBT_extra.csv
python -m industrial_ai.generate_extra --family ic --count 10000 --seed 103 --output data/generated/IC_extra.csv

