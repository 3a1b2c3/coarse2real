#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT=""
for candidate in \
  "${C2R_REPO_DIR:-}" \
  "${SLURM_SUBMIT_DIR:-}" \
  "${PWD}" \
  "${SCRIPT_DIR}/.."; do
  if [[ -n "${candidate}" && -f "${candidate}/inference/gradio_app.py" ]]; then
    REPO_ROOT="$(cd "${candidate}" && pwd)"
    break
  fi
done

if [[ -z "${REPO_ROOT}" ]]; then
  echo "Could not resolve the C2R repo root. Run from the repo root or set C2R_REPO_DIR=/path/to/coarse2real." >&2
  exit 1
fi

export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-${REPO_ROOT}/.gradio_tmp}"
mkdir -p "${GRADIO_TEMP_DIR}"
cd "${REPO_ROOT}"

python -m inference.gradio_app \
  --config inference/config_multigpu_usp.json \
  --examples-dir inference/control_videos \
  --server-name "${GRADIO_SERVER_NAME:-127.0.0.1}" \
  --server-port "${GRADIO_SERVER_PORT:-7860}"
