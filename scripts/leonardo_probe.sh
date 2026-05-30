#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "User: ${USER}"
echo "Scratch: ${SCRATCH:-unset}"
echo "Public: ${PUBLIC:-unset}"
python --version || true
which python || true
module list || true
sinfo -p "${HPC_SLURM_PARTITION:-boost_usr_prod}" || true

