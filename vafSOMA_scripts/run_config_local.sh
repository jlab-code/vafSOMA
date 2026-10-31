#!/usr/bin/env bash
set -euo pipefail
# Run one vafSOMA configuration with the project-local environment (built on first use).
#   bash vafSOMA_scripts/run_config_local.sh vafSOMA_configs/example_01_per_leaf.json [--outdir DIR]
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?provide a configuration file}"; shift
VAFSOMA_PYTHON="${VAFSOMA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${VAFSOMA_PYTHON}" ]]; then
  echo "Local .venv not found; building it." >&2
  bash "${PROJECT_ROOT}/vafSOMA_scripts/01_setup_env.sh"
  VAFSOMA_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${VAFSOMA_PYTHON}" -m vafsoma.cli check "${CONFIG}"
"${VAFSOMA_PYTHON}" -m vafsoma.cli run "${CONFIG}" "$@"
