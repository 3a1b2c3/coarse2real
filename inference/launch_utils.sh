#!/usr/bin/env bash

c2r_count_gpu_list() {
  local value="${1:-}"
  local count=0
  local token=""
  local -a tokens=()

  value="${value//[[:space:]]/}"
  if [[ -z "${value}" || "${value}" == "NoDevFiles" || "${value}" == "-1" ]]; then
    printf '0\n'
    return 0
  fi

  IFS=',' read -r -a tokens <<< "${value}"
  for token in "${tokens[@]}"; do
    [[ -z "${token}" ]] && continue
    if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      if (( end >= start )); then
        count=$((count + end - start + 1))
      else
        count=$((count + start - end + 1))
      fi
    else
      count=$((count + 1))
    fi
  done

  printf '%s\n' "${count}"
}

c2r_expand_gpu_list() {
  local value="${1:-}"
  local token=""
  local -a tokens=()
  local -a expanded=()

  value="${value//[[:space:]]/}"
  IFS=',' read -r -a tokens <<< "${value}"
  for token in "${tokens[@]}"; do
    [[ -z "${token}" ]] && continue
    if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      local idx
      if (( end >= start )); then
        for ((idx=start; idx<=end; idx++)); do
          expanded+=("${idx}")
        done
      else
        for ((idx=start; idx>=end; idx--)); do
          expanded+=("${idx}")
        done
      fi
    else
      expanded+=("${token}")
    fi
  done

  local IFS=','
  printf '%s\n' "${expanded[*]}"
}

c2r_numeric_gpu_count() {
  local value="${1:-}"
  value="${value//[[:space:]]/}"

  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  if [[ "${value}" =~ :([0-9]+)(\(|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  printf '0\n'
}

c2r_torch_cuda_device_count() {
  local python_bin="${PYTHON:-python}"
  command -v "${python_bin}" >/dev/null 2>&1 || {
    printf '0\n'
    return 0
  }

  "${python_bin}" - <<'PY' 2>/dev/null || printf '0\n'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
}

c2r_nvidia_smi_device_count() {
  local nvidia_output=""
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf '0\n'
    return 0
  }
  nvidia_output="$(nvidia-smi -L 2>/dev/null || true)"
  if [[ -z "${nvidia_output}" ]]; then
    printf '0\n'
    return 0
  fi
  printf '%s\n' "${nvidia_output}" | wc -l | tr -d '[:space:]'
  printf '\n'
}

c2r_maybe_export_scheduler_cuda_visible_devices() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    return 0
  fi

  local var=""
  local value=""
  local expanded=""
  local count=0
  for var in SLURM_STEP_GPUS SLURM_JOB_GPUS; do
    value="${!var:-}"
    [[ -z "${value}" ]] && continue
    # Resource descriptors such as gpu:a100:8 are counts, not CUDA device lists.
    # Bracketed/parenthesized Slurm strings are not valid CUDA_VISIBLE_DEVICES values.
    [[ "${value}" == *:* || "${value}" == *"["* || "${value}" == *"]"* || "${value}" == *"("* || "${value}" == *")"* ]] && continue
    expanded="$(c2r_expand_gpu_list "${value}")"
    count="$(c2r_count_gpu_list "${expanded}")"
    if (( count > 0 )); then
      export CUDA_VISIBLE_DEVICES="${expanded}"
      echo "CUDA_VISIBLE_DEVICES was not set; using ${var}=${expanded}."
      return 0
    fi
  done
}

c2r_detect_nproc_per_node() {
  local fallback="${1:-1}"
  local count=0
  local var=""
  local value=""

  c2r_maybe_export_scheduler_cuda_visible_devices

  count="$(c2r_count_gpu_list "${CUDA_VISIBLE_DEVICES:-}")"
  if (( count > 0 )); then
    C2R_DETECTED_NPROC_PER_NODE="${count}"
    C2R_NPROC_SOURCE="CUDA_VISIBLE_DEVICES"
    return 0
  fi

  for var in SLURM_GPUS_ON_NODE SLURM_GPUS_PER_NODE PBS_NUM_GPUS; do
    value="${!var:-}"
    [[ -z "${value}" ]] && continue
    count="$(c2r_numeric_gpu_count "${value}")"
    if (( count > 0 )); then
      C2R_DETECTED_NPROC_PER_NODE="${count}"
      C2R_NPROC_SOURCE="${var}"
      return 0
    fi
  done

  if [[ -n "${PBS_GPUFILE:-}" && -r "${PBS_GPUFILE}" ]]; then
    count="$(sort -u "${PBS_GPUFILE}" 2>/dev/null | wc -l | tr -d '[:space:]')"
    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      C2R_DETECTED_NPROC_PER_NODE="${count}"
      C2R_NPROC_SOURCE="PBS_GPUFILE"
      return 0
    fi
  fi

  count="$(c2r_torch_cuda_device_count)"
  if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
    C2R_DETECTED_NPROC_PER_NODE="${count}"
    C2R_NPROC_SOURCE="torch.cuda.device_count"
    return 0
  fi

  count="$(c2r_nvidia_smi_device_count)"
  if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
    C2R_DETECTED_NPROC_PER_NODE="${count}"
    C2R_NPROC_SOURCE="nvidia-smi"
    return 0
  fi

  C2R_DETECTED_NPROC_PER_NODE="${fallback}"
  C2R_NPROC_SOURCE="fallback"
}

c2r_configure_nproc_per_node() {
  local fallback="${1:-1}"

  c2r_maybe_export_scheduler_cuda_visible_devices

  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    export NPROC_PER_NODE
    echo "Using explicit NPROC_PER_NODE=${NPROC_PER_NODE}."
    return 0
  fi

  c2r_detect_nproc_per_node "${fallback}"
  NPROC_PER_NODE="${C2R_DETECTED_NPROC_PER_NODE}"
  export NPROC_PER_NODE
  echo "Using NPROC_PER_NODE=${NPROC_PER_NODE} (${C2R_NPROC_SOURCE})."
}

c2r_set_usp_timeout_defaults() {
  local nproc="${1:-1}"
  local heartbeat=120
  local pg_timeout=180
  local smoke_timeout=8
  local preflight_heartbeat=10
  local preflight_hard=20

  if (( nproc >= 8 )); then
    heartbeat=300
    pg_timeout=600
    smoke_timeout=60
    preflight_heartbeat=90
    preflight_hard=180
  elif (( nproc >= 4 )); then
    heartbeat=240
    pg_timeout=420
    smoke_timeout=30
    preflight_heartbeat=60
    preflight_hard=120
  fi

  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${heartbeat}}"
  export C2R_USP_PG_TIMEOUT_SEC="${C2R_USP_PG_TIMEOUT_SEC:-${pg_timeout}}"
  export C2R_NCCL_SMOKE_TIMEOUT_SEC="${C2R_NCCL_SMOKE_TIMEOUT_SEC:-${smoke_timeout}}"
  export C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC="${C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC:-${preflight_heartbeat}}"
  export C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC="${C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC:-${preflight_hard}}"
  export C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY="${C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY:-continue}"
}
