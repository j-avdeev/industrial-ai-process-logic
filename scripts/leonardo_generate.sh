#!/usr/bin/env bash
#SBATCH --job-name=indai-generate
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/slurm/generate-%j.log

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_common.sh"
mkdir -p artifacts/slurm data/generated

COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-50000}"
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
require_choice REQUIRE_SOURCE_BUNDLE "${REQUIRE_SOURCE_BUNDLE}" 0 1

READINESS_ARGS=(
  --count-per-family "${COUNT_PER_FAMILY}"
  --source-bundle "${SOURCE_BUNDLE}"
  --source-bundle-manifest "${SOURCE_BUNDLE_MANIFEST}"
  --out artifacts/leonardo_readiness.json
)
if [[ "${REQUIRE_SOURCE_BUNDLE}" == "1" ]]; then
  READINESS_ARGS+=(--require-source-bundle)
fi

python -m industrial_ai.leonardo_readiness "${READINESS_ARGS[@]}"
python -m industrial_ai.run_manifest \
  --stage generate_start \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.preflight --out artifacts/preflight_generate.json

python -m industrial_ai.generate_extra --family mosfet --count "${COUNT_PER_FAMILY}" --seed 101 --output data/generated/MOSFET_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.generate_extra --family igbt --count "${COUNT_PER_FAMILY}" --seed 102 --output data/generated/IGBT_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.generate_extra --family ic --count "${COUNT_PER_FAMILY}" --seed 103 --output data/generated/IC_extra.csv --skip-if-complete --exact-count
python -m industrial_ai.audit_corpus \
  --min-generated-per-family "${COUNT_PER_FAMILY}" \
  --max-generated-per-family "${COUNT_PER_FAMILY}"
python -m industrial_ai.prepare
python -m industrial_ai.run_manifest \
  --stage generate_complete \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
python -m industrial_ai.run_manifest \
  --stage generation_prepared \
  --set "COUNT_PER_FAMILY=${COUNT_PER_FAMILY}" \
  --set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"
