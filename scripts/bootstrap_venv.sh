#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

python3 --version
echo "[bootstrap] creating venv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel setuptools
if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing deps via uv"
  uv pip install -r requirements.txt
else
  echo "[bootstrap] installing deps via pip"
  pip install -r requirements.txt
fi

echo
echo "[bootstrap] done. activate with:"
echo "  source ${VENV_DIR}/bin/activate"
