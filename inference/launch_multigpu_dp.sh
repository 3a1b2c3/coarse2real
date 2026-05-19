#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_UTILS_PATH="${SCRIPT_DIR}/launch_utils.sh"
if [[ ! -f "${LAUNCH_UTILS_PATH}" ]]; then
  for candidate_dir in \
    "${C2R_INFERENCE_DIR:-}" \
    "${C2R_REPO_DIR:+${C2R_REPO_DIR}/inference}" \
    "${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/inference}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PBS_O_WORKDIR:+${PBS_O_WORKDIR}/inference}" \
    "${PBS_O_WORKDIR:-}" \
    "${LSB_SUBCWD:+${LSB_SUBCWD}/inference}" \
    "${LSB_SUBCWD:-}" \
    "${SGE_O_WORKDIR:+${SGE_O_WORKDIR}/inference}" \
    "${SGE_O_WORKDIR:-}" \
    "${PWD}/inference" \
    "${PWD}"; do
    if [[ -n "${candidate_dir}" && -f "${candidate_dir}/launch_utils.sh" ]]; then
      LAUNCH_UTILS_PATH="${candidate_dir}/launch_utils.sh"
      break
    fi
  done
fi
if [[ ! -f "${LAUNCH_UTILS_PATH}" ]]; then
  echo "Could not find launch_utils.sh. Run from the repo root, or set C2R_REPO_DIR=/path/to/repo or C2R_INFERENCE_DIR=/path/to/repo/inference." >&2
  exit 1
fi
source "${LAUNCH_UTILS_PATH}"

c2r_configure_nproc_per_node 4

CONFIG_PATH="${1:-${INFERENCE_CONFIG:-inference/config_multigpu_dp.json}}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m inference.run_inference \
  --config "${CONFIG_PATH}"
