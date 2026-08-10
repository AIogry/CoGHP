#!/usr/bin/env bash
set -euo pipefail

# Run the planner/executor comparison grid.
# 2026-8-6~8-7
# Configurations:
#   A: vanilla planner + matched-feedforward SwiGLU executor
#   B: FF SwiGLU planner + matched-feedforward SwiGLU executor
#   C: vanilla planner + HRM SwiGLU executor
#   D: FF SwiGLU planner + HRM SwiGLU executor
#   E: official HRM planner (H2L2/truncated or full) + matched-feedforward executor
#   F: official HRM planner (H2L2/truncated or full) + HRM SwiGLU executor
#   G: vanilla planner + original vanilla CoGHP executor
#   H: FF SwiGLU planner + original vanilla CoGHP executor
#   I: official HRM planner + original vanilla CoGHP executor
#   J: one shared official HRM actor core, truncated BPTT
#   K: one shared official HRM actor core, full BPTT
#
# C/D sweep H=2, L in {2,4,6}, with truncated and full BPTT.
# Every configuration is run on antmaze-giant and humanoidmaze-giant, seed 0.
#
# Local:
#   bash scripts/run_planner_executor_grid.sh plan
#   bash scripts/run_planner_executor_grid.sh launch
# Server:
#   SERVER_ENV=1 bash scripts/run_planner_executor_grid.sh plan
#   SERVER_ENV=1 bash scripts/run_planner_executor_grid.sh launch
#   SERVER_ENV=1 bash scripts/run_planner_executor_grid.sh launch_dynamic
#
# The server command should source env_xhumanoidserver.sh first (the launcher
# does this when SERVER_ENV=1).  DATA_ROOT/PYTHON can still be overridden.

MODE="${1:-plan}"
SERVER_ENV="${SERVER_ENV:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${HRM_ACTION_WANDB_PROJECT:-HRMActionCoGHPPlannerGrid}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${SERVER_ENV}" == "1" ]]; then
  # shellcheck source=scripts/env_xhumanoidserver.sh
  source "${PROJECT_ROOT}/scripts/env_xhumanoidserver.sh"
  DATA_ROOT="${DATA_ROOT:-/nix/data/offline-rl}"
  PYTHON="${PYTHON:-/root/Tools/miniforge3/envs/brain_nav/bin/python}"
  GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})
else
  DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
  PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"
GPU_IDS=(${GPU_IDS:-0 1})
fi
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
OUTPUT_ROOT="${HRM_ACTION_OUTPUT_ROOT:-${DATA_ROOT}/exp}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/logs/hrm_action_coghp_planner_grid}"
SELECTED_ONLY="${SELECTED_ONLY:-0}"
BEST_H_CYCLES="${BEST_H_CYCLES:-2}"
BEST_L_CYCLES="${BEST_L_CYCLES:-2}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
ACTION_MIXER_WIDTH="${ACTION_HRM_MIXER_INTERMEDIATE_SIZE:-256}"
GPU_FREE_MEMORY_MIB="${GPU_FREE_MEMORY_MIB:-1000}"
GPU_FREE_UTILIZATION="${GPU_FREE_UTILIZATION:-10}"

for value in "${BEST_H_CYCLES}" "${BEST_L_CYCLES}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BEST_H_CYCLES and BEST_L_CYCLES must be positive." >&2
    exit 2
  fi
done

export DATA_ROOT DATASET_DIR PYTHON
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/impls:${PYTHONPATH:-}"

JOB_LABELS=()
JOB_TASKS=()
JOB_PLANNERS=()
JOB_ACTIONS=()
JOB_H=()
JOB_L=()
JOB_BPTT=()
JOB_EXPECTED_PLANNER=()
JOB_PLANNER_H=()
JOB_PLANNER_L=()
JOB_PLANNER_BPTT=()
JOB_SHARED=()

add_job() {
  JOB_LABELS+=("$1")
  JOB_TASKS+=("$2")
  JOB_PLANNERS+=("$3")
  JOB_ACTIONS+=("$4")
  JOB_H+=("$5")
  JOB_L+=("$6")
  JOB_BPTT+=("$7")
  JOB_EXPECTED_PLANNER+=("$8")
  JOB_PLANNER_H+=("${9:-2}")
  JOB_PLANNER_L+=("${10:-2}")
  JOB_PLANNER_BPTT+=("${11:-True}")
  JOB_SHARED+=("${12:-False}")
}

for task in antmaze_giant humanoidmaze_giant; do
  if [[ "${task}" == "antmaze_giant" ]]; then
    task_short=antg
  else
    task_short=humg
  fi
  # The full task name and action settings already live in RUN_GROUP.  Keep
  # RUN_TAG compact so the W&B 128-character group limit is never exceeded.
  add_job "A_vanP_FFExec_${task_short}_s0" "${task}" vanilla_coghp matched_feedforward 2 2 True -1
  add_job "B_ffP_FFExec_${task_short}_s0" "${task}" matched_feedforward matched_feedforward 2 2 True 204288
  add_job "G_vanP_VanExec_${task_short}_s0" "${task}" vanilla_coghp vanilla_coghp 1 1 True -1
  add_job "H_ffP_VanExec_${task_short}_s0" "${task}" matched_feedforward vanilla_coghp 1 1 True 204288
  for planner in vanilla_coghp matched_feedforward; do
    if [[ "${planner}" == "vanilla_coghp" ]]; then
      planner_tag="C_vanillaPlanner"
    else
      planner_tag="D_FFPlanner"
    fi
    for l_cycles in 2 4 6; do
      for bptt in False True; do
        if [[ "${bptt}" == "True" ]]; then
          bptt_tag=fullbptt
        else
          bptt_tag=truncated
        fi
        add_job "${planner_tag}_HRMExec_h2l${l_cycles}_${bptt_tag}_${task_short}_s0" \
          "${task}" "${planner}" hrm 2 "${l_cycles}" "${bptt}" \
          "$([[ "${planner}" == "matched_feedforward" ]] && echo 204288 || echo -1)"
      done
    done
  done

  # Official HRM planner rows.  Planner H/L is fixed at H2/L2 while planner
  # BPTT is crossed with the executor schedule.  This explicitly includes
  # the planner-truncated condition requested for the comparison.
  for planner_bptt in False True; do
    if [[ "${planner_bptt}" == "True" ]]; then
      planner_bptt_tag=p_full
    else
      planner_bptt_tag=p_trunc
    fi
    add_job "E_hrmP_FFExec_ph2l2_${planner_bptt_tag}_${task_short}_s0" \
      "${task}" hrm matched_feedforward 1 1 True 204288 2 2 "${planner_bptt}"
    add_job "I_hrmP_VanExec_ph2l2_${planner_bptt_tag}_${task_short}_s0" \
      "${task}" hrm vanilla_coghp 1 1 True 204288 2 2 "${planner_bptt}"
    for l_cycles in 2 4 6; do
      for bptt in False True; do
        if [[ "${bptt}" == "True" ]]; then
          bptt_tag=fullbptt
          executor_bptt_tag=e_full
        else
          bptt_tag=truncated
          executor_bptt_tag=e_trunc
        fi
        add_job "F_hrmP_HRMExec_ph2l2_${planner_bptt_tag}_h2l${l_cycles}_${executor_bptt_tag}_${task_short}_s0" \
          "${task}" hrm hrm 2 "${l_cycles}" "${bptt}" 204288 2 2 "${planner_bptt}"
      done
    done
  done

  # Direct CoGHP-style comparison: planner and executor invoke one shared
  # official HRM parameter tree, while high/low readouts stay separate.
  add_job "J_sharedHRM_h${BEST_H_CYCLES}l${BEST_L_CYCLES}_truncated_${task_short}_s0" \
    "${task}" hrm hrm "${BEST_H_CYCLES}" "${BEST_L_CYCLES}" False 204288 "${BEST_H_CYCLES}" "${BEST_L_CYCLES}" False True
  add_job "K_sharedHRM_h${BEST_H_CYCLES}l${BEST_L_CYCLES}_fullbptt_${task_short}_s0" \
    "${task}" hrm hrm "${BEST_H_CYCLES}" "${BEST_L_CYCLES}" True 204288 "${BEST_H_CYCLES}" "${BEST_L_CYCLES}" True True
done

if (( ${#JOB_LABELS[@]} != 68 )); then
  echo "Internal grid error: expected 68 jobs, got ${#JOB_LABELS[@]}." >&2
  exit 2
fi

job_selected() {
  local index="$1"
  if [[ "${SELECTED_ONLY}" != "1" ]]; then
    return 0
  fi
  local label="${JOB_LABELS[index]}"
  case "${label}" in
    E_*|I_*|J_*|K_*) return 0 ;;
    F_*_h2l${BEST_L_CYCLES}_*) return 0 ;;
    *) return 1 ;;
  esac
}

print_plan() {
  local selected_count=0
  local count_index
  for count_index in "${!JOB_LABELS[@]}"; do
    if job_selected "${count_index}"; then
      selected_count=$((selected_count + 1))
    fi
  done
  echo "Planner/executor grid: ${selected_count} selected / ${#JOB_LABELS[@]} total jobs"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DATASET_DIR=${DATASET_DIR}"
  echo "PYTHON=${PYTHON}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "GPU_IDS=${GPU_IDS[*]}"
  echo "A/B: no HRM sweep; C/D: H=2, L=2/4/6, truncated/fullbptt"
  echo "E/F: HRM planner H=2,L=2, truncated/fullbptt; F also sweeps HRM executor H=2,L=2/4/6, truncated/fullbptt"
  echo "G/H/I: original vanilla CoGHP executor; I crosses planner H2L2 truncated/fullbptt"
  echo "J/K: one shared official HRM actor H=${BEST_H_CYCLES},L=${BEST_L_CYCLES}, truncated/fullbptt"
  echo "SELECTED_ONLY=${SELECTED_ONLY}; BEST_H_CYCLES=${BEST_H_CYCLES}; BEST_L_CYCLES=${BEST_L_CYCLES}"
  printf '%-4s %-72s\n' ID JOB
  local index
  for index in "${!JOB_LABELS[@]}"; do
    if ! job_selected "${index}"; then
      continue
    fi
    printf '%-4s %-72s\n' "${index}" "${JOB_LABELS[index]}"
  done
}

run_one() {
  local index="$1"
  local gpu="$2"
  local label="${JOB_LABELS[index]}"
  local task="${JOB_TASKS[index]}"
  local planner="${JOB_PLANNERS[index]}"
  local action="${JOB_ACTIONS[index]}"
  local h_cycles="${JOB_H[index]}"
  local l_cycles="${JOB_L[index]}"
  local bptt="${JOB_BPTT[index]}"
  local expected_planner="${JOB_EXPECTED_PLANNER[index]}"
  local planner_h_cycles="${JOB_PLANNER_H[index]}"
  local planner_l_cycles="${JOB_PLANNER_L[index]}"
  local planner_bptt="${JOB_PLANNER_BPTT[index]}"
  local shared_actor_core="${JOB_SHARED[index]}"
  local action_block=swiglu_rmsnorm
  local expected_action=204288
  if [[ "${action}" == "vanilla_coghp" ]]; then
    action_block=coghp_gelu
    expected_action=8734
  fi
  local log_file="${LOG_ROOT}/${label}.log"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "${LOG_ROOT}"
  fi
  echo "[$(date '+%F %T')] START ${label} GPU=${gpu} log=${log_file}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  DRY_RUN"
    return 0
  fi

  env \
    DATA_ROOT="${DATA_ROOT}" \
    DATASET_DIR="${DATASET_DIR}" \
    PYTHON="${PYTHON}" \
    HRM_ACTION_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HRM_ACTION_WANDB_PROJECT="${WANDB_PROJECT}" \
    LOG_DIR="${LOG_ROOT}" \
    WANDB_MODE="${WANDB_MODE}" \
    SEED=0 GPU_ID="${gpu}" \
    PLANNER_CORE_TYPE="${planner}" \
    PLANNER_BLOCK_TYPE=swiglu_rmsnorm \
    PLANNER_NUM_LAYERS=1 \
    PLANNER_MIXER_INTERMEDIATE_SIZE=256 \
    PLANNER_H_CYCLES="${planner_h_cycles}" \
    PLANNER_L_CYCLES="${planner_l_cycles}" \
    PLANNER_BPTT="${planner_bptt}" \
    EXPECTED_PLANNER_CORE_PARAMS="${expected_planner}" \
    ACTION_HRM_BACKBONE=mixer \
    ACTION_CORE_TYPE="${action}" \
    ACTION_BLOCK_TYPE="${action_block}" \
    HRM_BLOCK_TYPE="${action_block}" \
    SHARE_ACTOR_CORE="${shared_actor_core}" \
    ACTION_HRM_NUM_LAYERS=1 \
    ACTION_HRM_H_CYCLES="${h_cycles}" \
    ACTION_HRM_L_CYCLES="${l_cycles}" \
    ACTION_HRM_BPTT="${bptt}" \
    ACTION_HRM_MIXER_INTERMEDIATE_SIZE="${ACTION_MIXER_WIDTH}" \
    HRM_MIXER_INTERMEDIATE_SIZE="${ACTION_MIXER_WIDTH}" \
    ACTION_HRM_INTERMEDIATE_SIZE=512 \
    ACTION_HRM_HEAD_DIM=32 \
    ACTION_CHAIN_SOURCE=teacher \
    EXPECTED_ACTION_CORE_PARAMS="${expected_action}" \
    RUN_TAG="${label}" \
    bash "${PROJECT_ROOT}/scripts/train_hrm_action_coghp.sh" "${task}" \
    >"${log_file}" 2>&1
  echo "[$(date '+%F %T')] DONE ${label}"
}

run_queue() {
  local queue_id="$1"
  local index
  for index in "${!JOB_LABELS[@]}"; do
    if ! job_selected "${index}"; then
      continue
    fi
    if (( index % ${#GPU_IDS[@]} == queue_id )); then
      run_one "${index}" "${GPU_IDS[queue_id]}"
    fi
  done
}

# Dynamically assign the next job to whichever requested GPU is currently
# idle.  This is intentionally based on both memory and utilization: several
# users' launchers keep a small CUDA context alive after a job has finished,
# so a zero-context test alone is too brittle.  A reserved GPU is held until
# its child exits, preventing two jobs from racing during CUDA initialization.
gpu_is_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v target="${gpu}" \
        -v max_mem="${GPU_FREE_MEMORY_MIB}" \
        -v max_util="${GPU_FREE_UTILIZATION}" '
      $1 + 0 == target {
        gsub(/[[:space:]]/, "", $2)
        gsub(/[[:space:]]/, "", $3)
        found = 1
        ok = (($2 + 0) < max_mem && ($3 + 0) < max_util)
      }
      END { exit !(found && ok) }
    '
}

run_dynamic_queue() {
  local next_index=0
  local status=0
  local gpu
  local pid
  local index
  local -a active_pids=()
  declare -A reserved_gpu=()
  declare -A pid_gpu=()

  while (( next_index < ${#JOB_LABELS[@]} || ${#active_pids[@]} > 0 )); do
    local -a still_running=()
    for pid in "${active_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        still_running+=("${pid}")
      else
        if ! wait "${pid}"; then
          status=1
        fi
        unset "reserved_gpu[${pid_gpu[${pid}]}]"
        unset "pid_gpu[${pid}]"
      fi
    done
    active_pids=("${still_running[@]}")

    while (( next_index < ${#JOB_LABELS[@]} )); do
      if ! job_selected "${next_index}"; then
        next_index=$((next_index + 1))
        continue
      fi
      gpu=""
      for index in "${GPU_IDS[@]}"; do
        if [[ "${reserved_gpu[${index}]:-0}" == "1" ]]; then
          continue
        fi
        if gpu_is_idle "${index}"; then
          gpu="${index}"
          break
        fi
      done
      if [[ -z "${gpu}" ]]; then
        break
      fi
      run_one "${next_index}" "${gpu}" &
      pid=$!
      active_pids+=("${pid}")
      reserved_gpu[${gpu}]=1
      pid_gpu[${pid}]="${gpu}"
      next_index=$((next_index + 1))
    done

    if (( next_index < ${#JOB_LABELS[@]} || ${#active_pids[@]} > 0 )); then
      sleep "${GPU_POLL_INTERVAL_SECONDS:-30}"
    fi
  done
  return "${status}"
}

case "${MODE}" in
  plan)
    print_plan
    ;;
  launch)
    print_plan
    echo "Launching ${#GPU_IDS[@]} queue(s)."
    queue_pids=()
    for queue_id in "${!GPU_IDS[@]}"; do
      run_queue "${queue_id}" &
      queue_pids+=("$!")
    done
    status=0
    for pid in "${queue_pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if (( status != 0 )); then
      echo "At least one planner/executor queue failed; inspect ${LOG_ROOT}." >&2
      exit "${status}"
    fi
    echo "All planner/executor grid jobs completed."
    ;;
  launch_dynamic)
    print_plan
    echo "Launching dynamic GPU scheduler on GPU_IDS=${GPU_IDS[*]} (free-memory threshold=${GPU_FREE_MEMORY_MIB} MiB, utilization threshold=${GPU_FREE_UTILIZATION}%)."
    if ! run_dynamic_queue; then
      echo "At least one dynamically scheduled job failed; inspect ${LOG_ROOT}." >&2
      exit 1
    fi
    echo "All dynamically scheduled planner/executor jobs completed."
    ;;
  *)
    echo "Usage: $0 {plan|launch|launch_dynamic}" >&2
    exit 2
    ;;
esac
