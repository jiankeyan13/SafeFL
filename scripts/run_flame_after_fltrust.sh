#!/usr/bin/env bash
# 等待 FLTrust 批次 (5 个实验) 全部结束后, 启动 FLAME 批次.
#
# 用法:
#   nohup bash scripts/run_flame_after_fltrust.sh > logs/batch_runs/flame_after_fltrust.log 2>&1 &

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/logs/batch_runs"
SCHEDULER_LOG="${LOG_DIR}/flame_after_fltrust.log"
POOL_LOG="${LOG_DIR}/flame_pool.log"
PROJECT="${WANDB_PROJECT:-SafeFL_Cifar10_q0.5}"
ACTORS_PER_GPU="${ACTORS_PER_GPU:-10}"
NUM_GPUS="${NUM_GPUS:-4}"
POLL_SEC="${POLL_SEC:-60}"

FLTRUST_EXPS=(vanilla_fltrust dba_fltrust neurotoxin_fltrust pgd_fltrust lga_fltrust)
FLAME_EXPS=(vanilla_flame dba_flame neurotoxin_flame pgd_flame lga_flame)

mkdir -p "${LOG_DIR}"

log() {
  local msg="[$(date '+%F %T')] $*"
  echo "${msg}" | tee -a "${SCHEDULER_LOG}"
}

log_pool() {
  local msg="[$(date '+%F %T')] $*"
  echo "${msg}" >> "${POOL_LOG}"
}

wait_for_fltrust() {
  log "Waiting for FLTrust batch: ${FLTRUST_EXPS[*]}"
  while true; do
    if pgrep -f "main.py experiment=.*_fltrust" >/dev/null 2>&1; then
      local running
      running="$(pgrep -af "main.py experiment=.*_fltrust" 2>/dev/null | grep -oE 'experiment=[^ ]+' | sort -u | tr '\n' ' ' || true)"
      log "Still running: ${running:-unknown}"
      sleep "${POLL_SEC}"
      continue
    fi
    if [[ ! -f "${LOG_DIR}/lga_fltrust.log" ]]; then
      log "lga_fltrust not started yet, keep waiting..."
      sleep "${POLL_SEC}"
      continue
    fi
    log "FLTrust batch complete."
    break
  done
}

run_experiment() {
  local exp="$1"
  local gpu="$2"
  local logfile="${LOG_DIR}/${exp}.log"

  log_pool "[GPU ${gpu}] START ${exp}"
  cd "${ROOT}"
  if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" main.py \
    "experiment=${exp}" \
    "experiment_name=${exp}" \
    "logging.name=${exp}" \
    "logging.project=${PROJECT}" \
    "parallel.gpu_ids=[${gpu}]" \
    "parallel.actors_per_gpu=${ACTORS_PER_GPU}" \
    > "${logfile}" 2>&1; then
    log_pool "[GPU ${gpu}] DONE  ${exp}"
    return 0
  else
    log_pool "[GPU ${gpu}] FAIL  ${exp}, see ${logfile}"
    return 1
  fi
}

run_flame_pool() {
  log "Starting FLAME batch: ${FLAME_EXPS[*]}"
  echo "=== FLAME pool started at $(date '+%F %T') ===" >> "${POOL_LOG}"

  local fifo
  fifo="$(mktemp -u)"
  mkfifo "${fifo}"
  exec 3<>"${fifo}"
  rm -f "${fifo}"

  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    echo "${gpu}" >&3
  done

  local pids=() ok=0 fail=0
  for exp in "${FLAME_EXPS[@]}"; do
    read -r gpu <&3
    (
      if run_experiment "${exp}" "${gpu}"; then
        echo "${gpu}" >&3
        exit 0
      else
        echo "${gpu}" >&3
        exit 1
      fi
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      ((ok++)) || true
    else
      ((fail++)) || true
    fi
  done

  exec 3>&-
  log "FLAME batch finished (ok=${ok}, fail=${fail})"
  echo "=== FLAME pool finished at $(date '+%F %T') (ok=${ok}, fail=${fail}) ===" >> "${POOL_LOG}"
}

main() {
  log "Scheduler started."
  wait_for_fltrust
  run_flame_pool
  log "Scheduler finished."
}

main "$@"
