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

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-20000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

manual_nccl_override=0
if [[ -n "${NCCL_P2P_DISABLE:-}" || -n "${NCCL_P2P_LEVEL:-}" || -n "${NCCL_SHM_DISABLE:-}" ]]; then
  manual_nccl_override=1
fi

if [[ "${C2R_NCCL_SAFE_MODE:-0}" == "1" ]]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  manual_nccl_override=1
fi

if [[ "${C2R_NCCL_SHM_FALLBACK:-0}" == "1" ]]; then
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  manual_nccl_override=1
fi

c2r_configure_nproc_per_node 4
c2r_set_usp_timeout_defaults "${NPROC_PER_NODE}"

CONFIG_PATH="${1:-${INFERENCE_CONFIG:-inference/config_multigpu_usp.json}}"

runtime_root="${C2R_RUNTIME_DIR:-${TMPDIR:-/tmp}}/c2r_runtime"
nccl_mode_cache_dir="${runtime_root}/nccl_transport_cache"
mkdir -p "${nccl_mode_cache_dir}"

reset_nccl_transport_overrides() {
  unset NCCL_P2P_DISABLE
  unset NCCL_P2P_LEVEL
  unset NCCL_SHM_DISABLE
}

apply_nccl_transport_mode() {
  local mode="$1"
  reset_nccl_transport_overrides
  case "${mode}" in
    default)
      ;;
    pix)
      export NCCL_P2P_LEVEL=PIX
      ;;
    safe)
      export NCCL_P2P_DISABLE=1
      ;;
    *)
      echo "Unknown NCCL transport mode: ${mode}" >&2
      return 1
      ;;
  esac
}

build_nccl_cache_file() {
  local host_id
  local gpu_id
  local key_source
  local key_hash
  host_id="$(hostname 2>/dev/null || uname -n || echo unknown-host)"
  gpu_id="${CUDA_VISIBLE_DEVICES:-all}"
  key_source="${host_id}|${gpu_id}|${NPROC_PER_NODE}"
  if command -v sha1sum >/dev/null 2>&1; then
    key_hash="$(printf '%s' "${key_source}" | sha1sum | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    key_hash="$(printf '%s' "${key_source}" | shasum | awk '{print $1}')"
  else
    key_hash="$(printf '%s' "${key_source}" | tr -c '[:alnum:]._-' '_')"
  fi
  printf '%s/%s.env\n' "${nccl_mode_cache_dir}" "${key_hash}"
}

nccl_mode_cache_file="$(build_nccl_cache_file)"

save_cached_nccl_mode() {
  local label="$1"
  cat > "${nccl_mode_cache_file}" <<EOF
LABEL=${label}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-}
NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-}
EOF
}

load_cached_nccl_mode() {
  local label=""
  local key=""
  local value=""
  if [[ ! -f "${nccl_mode_cache_file}" ]]; then
    return 1
  fi

  reset_nccl_transport_overrides
  while IFS='=' read -r key value; do
    case "${key}" in
      LABEL)
        label="${value}"
        ;;
      NCCL_P2P_DISABLE)
        if [[ -n "${value}" ]]; then
          export NCCL_P2P_DISABLE="${value}"
        fi
        ;;
      NCCL_P2P_LEVEL)
        if [[ -n "${value}" ]]; then
          export NCCL_P2P_LEVEL="${value}"
        fi
        ;;
      NCCL_SHM_DISABLE)
        if [[ -n "${value}" ]]; then
          export NCCL_SHM_DISABLE="${value}"
        fi
        ;;
    esac
  done < "${nccl_mode_cache_file}"

  if [[ -z "${label}" ]]; then
    rm -f "${nccl_mode_cache_file}"
    return 1
  fi

  C2R_CACHED_NCCL_LABEL="${label}"
  return 0
}

run_nccl_preflight() {
  local label="$1"
  local preflight_log=""
  local status=0
  local timed_out=0
  C2R_LAST_NCCL_PREFLIGHT_RESULT="unknown"
  echo "Running NCCL preflight (${label}) with NPROC_PER_NODE=${NPROC_PER_NODE}..."
  echo "NCCL transport overrides: NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-auto} NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}"
  preflight_log="$(mktemp)"
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=2s "${C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC}s" \
      env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC}" \
      C2R_NCCL_SMOKE_TIMEOUT_SEC="${C2R_NCCL_SMOKE_TIMEOUT_SEC}" \
      torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m inference.nccl_smoke_test \
      >"${preflight_log}" 2>&1
    status=$?
  else
    env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC}" \
      C2R_NCCL_SMOKE_TIMEOUT_SEC="${C2R_NCCL_SMOKE_TIMEOUT_SEC}" \
      torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m inference.nccl_smoke_test \
      >"${preflight_log}" 2>&1
    status=$?
  fi

  if [[ "${status}" -eq 0 ]]; then
    C2R_LAST_NCCL_PREFLIGHT_RESULT="passed"
    grep -E "Running NCCL preflight on|NCCL preflight passed\\." "${preflight_log}" || true
    rm -f "${preflight_log}"
    return 0
  fi

  if [[ "${status}" -eq 124 || "${status}" -eq 143 || "${status}" -eq 137 ]]; then
    timed_out=1
  elif grep -qiE "timed out|timeout|Watchdog" "${preflight_log}"; then
    timed_out=1
  fi

  if [[ "${timed_out}" -eq 1 ]]; then
    C2R_LAST_NCCL_PREFLIGHT_RESULT="timeout"
    echo "NCCL preflight (${label}) timed out after ${C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC}s. Trying next transport mode..."
  else
    C2R_LAST_NCCL_PREFLIGHT_RESULT="failed"
    echo "NCCL preflight (${label}) failed with exit code ${status}. Trying next transport mode..."
  fi

  grep -E "Running NCCL preflight on|Watchdog|timed out|failed|ProcessGroupNCCL|Warning" "${preflight_log}" \
    | grep -v "Traceback" \
    | grep -v "torch.distributed.elastic.multiprocessing.api.SignalException" \
    || true

  rm -f "${preflight_log}"
  return "${status}"
}

if [[ "${C2R_SKIP_NCCL_PREFLIGHT:-0}" != "1" ]]; then
  if [[ "${manual_nccl_override}" == "1" ]]; then
    set +e
    run_nccl_preflight "manual overrides"
    preflight_status=$?
    preflight_result="${C2R_LAST_NCCL_PREFLIGHT_RESULT}"
    set -e
    if [[ "${preflight_status}" -ne 0 ]]; then
      if [[ "${preflight_result}" == "timeout" && "${C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY}" == "continue" ]]; then
        echo "NCCL preflight timed out with manual transport overrides; continuing because C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=continue."
      else
        exit "${preflight_status}"
      fi
    fi
  else
    set +e

    selected_transport=""
    preflight_status=1
    preflight_timeout_seen=0
    preflight_non_timeout_failure=0

    if load_cached_nccl_mode; then
      echo "Trying cached NCCL transport mode: ${C2R_CACHED_NCCL_LABEL}"
      run_nccl_preflight "${C2R_CACHED_NCCL_LABEL}"
      preflight_status=$?
      if [[ "${preflight_status}" -eq 0 ]]; then
        selected_transport="${C2R_CACHED_NCCL_LABEL}"
      else
        if [[ "${C2R_LAST_NCCL_PREFLIGHT_RESULT}" == "timeout" ]]; then
          preflight_timeout_seen=1
        else
          preflight_non_timeout_failure=1
        fi
        rm -f "${nccl_mode_cache_file}"
      fi
    fi

    if [[ -z "${selected_transport}" ]]; then
      apply_nccl_transport_mode default
      run_nccl_preflight "default"
      preflight_status=$?
      if [[ "${preflight_status}" -eq 0 ]]; then
        selected_transport="default"
      else
        if [[ "${C2R_LAST_NCCL_PREFLIGHT_RESULT}" == "timeout" ]]; then
          preflight_timeout_seen=1
        else
          preflight_non_timeout_failure=1
        fi
        apply_nccl_transport_mode pix
        run_nccl_preflight "NCCL_P2P_LEVEL=PIX"
        preflight_status=$?
        if [[ "${preflight_status}" -eq 0 ]]; then
          selected_transport="NCCL_P2P_LEVEL=PIX"
        else
          if [[ "${C2R_LAST_NCCL_PREFLIGHT_RESULT}" == "timeout" ]]; then
            preflight_timeout_seen=1
          else
            preflight_non_timeout_failure=1
          fi
          apply_nccl_transport_mode safe
          run_nccl_preflight "NCCL_P2P_DISABLE=1"
          preflight_status=$?
          if [[ "${preflight_status}" -eq 0 ]]; then
            selected_transport="NCCL_P2P_DISABLE=1"
          else
            if [[ "${C2R_LAST_NCCL_PREFLIGHT_RESULT}" == "timeout" ]]; then
              preflight_timeout_seen=1
            else
              preflight_non_timeout_failure=1
            fi
          fi
        fi
      fi
    fi

    set -e

    if [[ "${preflight_status}" -ne 0 ]]; then
      if [[ "${preflight_timeout_seen}" -eq 1 && "${preflight_non_timeout_failure}" -eq 0 && "${C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY}" == "continue" ]]; then
        apply_nccl_transport_mode default
        echo "NCCL preflight timed out for all transport modes but did not report a concrete NCCL error."
        echo "Continuing with default NCCL transport because C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=continue."
        echo "Set C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=fail or C2R_SKIP_NCCL_PREFLIGHT=1 to change this behavior."
      else
        echo "NCCL preflight failed for default, NCCL_P2P_LEVEL=PIX, and NCCL_P2P_DISABLE=1." >&2
        exit "${preflight_status}"
      fi
    else
      save_cached_nccl_mode "${selected_transport}"
      echo "Selected NCCL transport mode: ${selected_transport}"
    fi
  fi
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m inference.run_inference \
  --config "${CONFIG_PATH}"
