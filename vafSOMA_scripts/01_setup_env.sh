#!/usr/bin/env bash
set -euo pipefail
# Build the project-local vafSOMA virtual environment (.venv) and install vafSOMA into it.
#   bash vafSOMA_scripts/01_setup_env.sh                 # uses python3 (>= 3.10)
#   PYTHON_BIN=/path/to/python3.11 bash vafSOMA_scripts/01_setup_env.sh
#   VAFSOMA_LOCKED=0 bash vafSOMA_scripts/01_setup_env.sh  # minimum versions (requirements.txt) instead of the lock
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
unset PYTHONPATH PYTHONHOME 2>/dev/null || true
export PYTHONNOUSERSITE=1

"${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "vafSOMA needs Python >= 3.10 (found $("${PYTHON_BIN}" --version 2>&1)); set PYTHON_BIN." >&2; exit 1; }
REQ_FILE="${PROJECT_ROOT}/requirements.lock.txt"
[[ "${VAFSOMA_LOCKED:-1}" == "1" && -f "${REQ_FILE}" ]] || REQ_FILE="${PROJECT_ROOT}/requirements.txt"

echo "Project root:      ${PROJECT_ROOT}"
echo "Python:            $("${PYTHON_BIN}" --version 2>&1)"
echo "Requirements file: ${REQ_FILE}"
[[ -x "${VENV_DIR}/bin/python" ]] || "${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"
"${VENV_DIR}/bin/python" -m pip install --no-deps -e "${PROJECT_ROOT}"
"${VENV_DIR}/bin/vafsoma" version
echo "Virtual environment ready: ${VENV_DIR}  (activate with: source .venv/bin/activate)"
