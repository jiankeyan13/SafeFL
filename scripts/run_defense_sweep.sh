#!/usr/bin/env bash
# 两阶段批量实验:
#
# Phase 1 - 4 攻击 x 9 防御 (36 个实验)
#   每张 GPU 固定一种攻击, 在该卡上串行跑 9 种防御.
#   GPU 0: vanilla  GPU 1: dba  GPU 2: neurotoxin  GPU 3: pgd
#
# Phase 2 - LGA x 11 防御 (11 个实验, 在 Phase 1 全部结束后启动)
#   4 张 GPU 共享任务池, 哪个空了就接下一个实验 (不绑定攻击到固定卡).
#   防御: vanilla, multikrum, trimmean, rfa, foolsgold, fltrust, dnc, rlr, flame, rflbat, alignins
#
# 用法:
#   bash scripts/run_defense_sweep.sh
#   nohup bash scripts/run_defense_sweep.sh > logs/defense_sweep/master.log 2>&1 &
#
# 约定:
#   - wandb 项目: SafeFL_Cifar10_q0.5
#   - 实验名: {attack}_{defense}
#   - CUDA_VISIBLE_DEVICES=N 与 parallel.gpu_ids=[N] 配对

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/logs/defense_sweep"
PROJECT="${WANDB_PROJECT:-SafeFL_Cifar10_q0.5}"
ACTORS_PER_GPU="${ACTORS_PER_GPU:-10}"
NUM_GPUS="${NUM_GPUS:-4}"

# Phase 1: 9 种防御 (不含 vanilla / multikrum)
DEFENSES=(
  trimmean
  rfa
  foolsgold
  fltrust
  dnc
  rlr
  flame
  rflbat
  alignins
)

declare -A ATTACK_GPU=(
  [vanilla]=0
  [dba]=1
  [neurotoxin]=2
  [pgd]=3
)

PHASE1_ATTACKS=(vanilla dba neurotoxin pgd)

# Phase 2: LGA + 无防御 + MultiKrum + 9 种防御
LGA_DEFENSES=(
  vanilla
  multikrum
  trimmean
  rfa
  foolsgold
  fltrust
  dnc
  rlr
  flame
  rflbat
  alignins
)

mkdir -p "${LOG_DIR}"

log_line() {
  local msg="$1"
  local file="${2:-${LOG_DIR}/master.log}"
  echo "${msg}" | tee -a "${file}"
}

run_experiment() {
  local exp="$1"
  local gpu="$2"
  local queue_log="${3:-${LOG_DIR}/gpu${gpu}.log}"
  local log="${LOG_DIR}/${exp}.log"

  if [[ ! -f "${ROOT}/configs/experiment/${exp}.yaml" ]]; then
    log_line "[GPU ${gpu}] SKIP ${exp}: missing configs/experiment/${exp}.yaml" "${queue_log}"
    return 1
  fi

  {
    echo "========================================"
    echo "[GPU ${gpu}] START ${exp} at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "CUDA_VISIBLE_DEVICES=${gpu} parallel.gpu_ids=[${gpu}] actors_per_gpu=${ACTORS_PER_GPU}"
  } | tee -a "${queue_log}"

  cd "${ROOT}"
  if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" main.py \
    "experiment=${exp}" \
    "experiment_name=${exp}" \
    "logging.name=${exp}" \
    "logging.project=${PROJECT}" \
    "parallel.gpu_ids=[${gpu}]" \
    "parallel.actors_per_gpu=${ACTORS_PER_GPU}" \
    > "${log}" 2>&1; then
    log_line "[GPU ${gpu}] DONE  ${exp} at $(date '+%Y-%m-%d %H:%M:%S')" "${queue_log}"
    return 0
  else
    local code=$?
    log_line "[GPU ${gpu}] FAIL  ${exp} at $(date '+%Y-%m-%d %H:%M:%S')} (exit ${code}), see ${log}" "${queue_log}"
    return "${code}"
  fi
}

run_attack_queue() {
  local attack="$1"
  local gpu="${ATTACK_GPU[$attack]}"
  local queue_log="${LOG_DIR}/phase1_gpu${gpu}_${attack}.log"
  local ok=0
  local fail=0

  echo "=== Phase 1 | GPU ${gpu} | ${attack} started at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${queue_log}"

  for defense in "${DEFENSES[@]}"; do
    local exp="${attack}_${defense}"
    if run_experiment "${exp}" "${gpu}" "${queue_log}"; then
      ((ok++)) || true
    else
      ((fail++)) || true
    fi
  done

  log_line "=== Phase 1 | GPU ${gpu} | ${attack} finished (ok=${ok}, fail=${fail}) ===" "${queue_log}"
}

run_phase1() {
  log_line "========== Phase 1 started at $(date '+%Y-%m-%d %H:%M:%S') =========="
  log_line "Attacks: ${PHASE1_ATTACKS[*]}"
  log_line "Defenses (${#DEFENSES[@]}): ${DEFENSES[*]}"
  log_line "Total: $((${#PHASE1_ATTACKS[@]} * ${#DEFENSES[@]})) experiments"

  local pids=()
  for attack in "${PHASE1_ATTACKS[@]}"; do
    run_attack_queue "${attack}" &
    pids+=($!)
    log_line "Launched Phase 1 queue: ${attack} on GPU ${ATTACK_GPU[$attack]} (pid $!)"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || ((failed++)) || true
  done

  log_line "========== Phase 1 finished at $(date '+%Y-%m-%d %H:%M:%S') (failed queues: ${failed}) =========="
  return "${failed}"
}

run_phase2() {
  local pool_log="${LOG_DIR}/phase2_lga_pool.log"
  local -a experiments=()
  local ok=0
  local fail=0

  for defense in "${LGA_DEFENSES[@]}"; do
    experiments+=("lga_${defense}")
  done

  log_line "========== Phase 2 started at $(date '+%Y-%m-%d %H:%M:%S') ==========" "${pool_log}"
  log_line "LGA defenses (${#experiments[@]}): ${experiments[*]}" "${pool_log}"
  log_line "GPU pool: 0-$((NUM_GPUS - 1)), one experiment per free GPU" "${pool_log}"

  # GPU 令牌池: 空闲 GPU 编号写入 fifo, worker 取走后归还
  local fifo
  fifo="$(mktemp -u)"
  mkfifo "${fifo}"
  exec 3<>"${fifo}"
  rm -f "${fifo}"

  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    echo "${gpu}" >&3
  done

  local pids=()
  for exp in "${experiments[@]}"; do
    read -r gpu <&3
    (
      if run_experiment "${exp}" "${gpu}" "${pool_log}"; then
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

  log_line "========== Phase 2 finished at $(date '+%Y-%m-%d %H:%M:%S')} (ok=${ok}, fail=${fail}) ==========" "${pool_log}"
  return "${fail}"
}

main() {
  log_line "Defense sweep started at $(date '+%Y-%m-%d %H:%M:%S')"
  log_line "Project: ${PROJECT} | Logs: ${LOG_DIR}"

  local phase1_fail=0
  local phase2_fail=0

  run_phase1 || phase1_fail=$?

  log_line "Phase 1 complete. Starting Phase 2 (LGA) ..."
  run_phase2 || phase2_fail=$?

  log_line "All phases finished at $(date '+%Y-%m-%d %H:%M:%S')"
  if (( phase1_fail > 0 || phase2_fail > 0 )); then
    log_line "Warning: phase1_failed_queues=${phase1_fail}, phase2_failed_experiments=${phase2_fail}"
    exit 1
  fi
}

main "$@"
