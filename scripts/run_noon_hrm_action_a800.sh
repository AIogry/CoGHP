#!/usr/bin/env bash
set -euo pipefail

# Sequential experiment queues for the 2026-08-04 A800 run.
#
# Usage on xhumanoidserver:
#   bash scripts/run_noon_hrm_action_a800.sh plan
#   bash scripts/run_noon_hrm_action_a800.sh hum_baseline_main 1
#
# Each queue owns exactly one physical GPU and runs its jobs sequentially.
# The launcher is expected to redirect the queue-level stdout/stderr to a
# tmux log.  Every individual training job also receives its own log below
# /nix/data/offline-rl/logs/hrm_action_noon_20260804.

QUEUE="${1:-plan}"
GPU_ID="${2:-}"
DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/env_xhumanoidserver.sh"

# Make the server paths explicit even if the calling shell exported stale
# local values before sourcing env_xhumanoidserver.sh.
DATA_ROOT="${A800_DATA_ROOT:-/nix/data/offline-rl}"
DATASET_DIR="${A800_DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
PYTHON="${A800_PYTHON:-/root/Tools/miniforge3/envs/brain_nav/bin/python}"
LOG_ROOT="${A800_LOG_ROOT:-${DATA_ROOT}/logs/hrm_action_noon_20260804}"

WANDB_MODE="${WANDB_MODE:-online}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_TAG="${RUN_TAG:-a800}"

export DATA_ROOT DATASET_DIR OGBENCH_DATASET_DIR="${DATASET_DIR}" PYTHON
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/impls:${PYTHONPATH:-}"

print_plan() {
  cat <<'EOF'
Queue                         GPU  Sequential jobs
hum_baseline_main               1  humanoid baseline s0 -> H1L1 s0
hum_trunc_h2l6                  2  humanoid H2L2 truncated s0 -> H2L6 full s0
hum_n2_attention                3  humanoid N2 H2L2 full s0 -> Attention H2L2 full s0
ant_baseline_trunc_seeds        4  ant baseline s1 -> baseline s2 -> H2L2 truncated s1 -> s2
ant_n2_seeds_and_trunc          5  ant N2 H2L2 full s1 -> s2 -> N2 H2L2 truncated s0
ant_h2l6_seeds_and_trunc        6  ant H2L6 full s1 -> s2 -> H2L6 truncated s0
hum_main_spare                  7  humanoid main H2L2 full s0

Notes:
  * GPU 0 is deliberately unused because it was occupied by a non-CoGHP job.
  * hum_main_spare is separated so the key full-vs-truncated Humanoid pair
    starts concurrently. If only seven GPUs remain free, launch all queues.
  * All jobs use batch size 256, 1M updates, teacher chains, temperature 0,
    evaluation every 100k updates, and 50 episodes per task.
EOF
}

require_gpu() {
  if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "A numeric GPU ID is required for queue ${QUEUE}." >&2
    exit 2
  fi
}

run_logged() {
  local label="$1"
  shift
  local log_file="${LOG_ROOT}/${label}.log"

  printf '[%s] START %s GPU=%s\n' "$(date '+%F %T')" "${label}" "${GPU_ID}"
  printf '  log=%s\n' "${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  mkdir -p "${LOG_ROOT}"
  if "$@" >"${log_file}" 2>&1; then
    printf '[%s] DONE  %s\n' "$(date '+%F %T')" "${label}"
  else
    local status=$?
    printf '[%s] FAIL  %s status=%s log=%s\n' \
      "$(date '+%F %T')" "${label}" "${status}" "${log_file}" >&2
    return "${status}"
  fi
}

run_baseline() {
  local task="$1"
  local seed="$2"
  local label="$3"

  run_logged "${label}" env \
    DATA_ROOT="${DATA_ROOT}" \
    DATASET_DIR="${DATASET_DIR}" \
    PYTHON="${PYTHON}" \
    SEED="${seed}" \
    LARGE_GPU="${GPU_ID}" \
    GIANT_GPU="${GPU_ID}" \
    WANDB_MODE="${WANDB_MODE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    RUN_SUFFIX=_split_action_baseline_a800 \
    bash "${PROJECT_ROOT}/scripts/train_coghp.sh" \
      "${task}_split_action_baseline"
}

run_hrm() {
  local task="$1"
  local seed="$2"
  local backbone="$3"
  local num_layers="$4"
  local h_cycles="$5"
  local l_cycles="$6"
  local bptt="$7"
  local mixer_width="$8"
  local label="$9"

  run_logged "${label}" env \
    DATA_ROOT="${DATA_ROOT}" \
    DATASET_DIR="${DATASET_DIR}" \
    PYTHON="${PYTHON}" \
    SEED="${seed}" \
    GPU_ID="${GPU_ID}" \
    WANDB_MODE="${WANDB_MODE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    RUN_TAG="${RUN_TAG}" \
    ACTION_HRM_BACKBONE="${backbone}" \
    ACTION_HRM_NUM_LAYERS="${num_layers}" \
    ACTION_HRM_H_CYCLES="${h_cycles}" \
    ACTION_HRM_L_CYCLES="${l_cycles}" \
    ACTION_HRM_BPTT="${bptt}" \
    ACTION_HRM_MIXER_INTERMEDIATE_SIZE="${mixer_width}" \
    ACTION_HRM_INTERMEDIATE_SIZE=512 \
    ACTION_HRM_HEAD_DIM=32 \
    ACTION_CHAIN_SOURCE=teacher \
    bash "${PROJECT_ROOT}/scripts/train_hrm_action_coghp.sh" "${task}"
}

case "${QUEUE}" in
  plan)
    print_plan
    ;;
  hum_baseline_main)
    require_gpu
    run_baseline humanoidmaze_giant 0 hm_baseline_s0
    run_hrm humanoidmaze_giant 0 mixer 1 1 1 True 256 hm_h1l1_full_m256_s0
    ;;
  hum_trunc_h2l6)
    require_gpu
    run_hrm humanoidmaze_giant 0 mixer 1 2 2 False 256 hm_h2l2_truncated_m256_s0
    run_hrm humanoidmaze_giant 0 mixer 1 2 6 True 256 hm_h2l6_full_m256_s0
    ;;
  hum_n2_attention)
    require_gpu
    run_hrm humanoidmaze_giant 0 mixer 2 2 2 True 256 hm_n2_h2l2_full_m256_s0
    run_hrm humanoidmaze_giant 0 attention 1 2 2 True 256 hm_attention_h2l2_full_s0
    ;;
  ant_baseline_trunc_seeds)
    require_gpu
    run_baseline antmaze_giant 1 ant_baseline_s1
    run_baseline antmaze_giant 2 ant_baseline_s2
    run_hrm antmaze_giant 1 mixer 1 2 2 False 256 ant_h2l2_truncated_m256_s1
    run_hrm antmaze_giant 2 mixer 1 2 2 False 256 ant_h2l2_truncated_m256_s2
    ;;
  ant_n2_seeds_and_trunc)
    require_gpu
    run_hrm antmaze_giant 1 mixer 2 2 2 True 256 ant_n2_h2l2_full_m256_s1
    run_hrm antmaze_giant 2 mixer 2 2 2 True 256 ant_n2_h2l2_full_m256_s2
    run_hrm antmaze_giant 0 mixer 2 2 2 False 256 ant_n2_h2l2_truncated_m256_s0
    ;;
  ant_h2l6_seeds_and_trunc)
    require_gpu
    run_hrm antmaze_giant 1 mixer 1 2 6 True 256 ant_h2l6_full_m256_s1
    run_hrm antmaze_giant 2 mixer 1 2 6 True 256 ant_h2l6_full_m256_s2
    run_hrm antmaze_giant 0 mixer 1 2 6 False 256 ant_h2l6_truncated_m256_s0
    ;;
  hum_main_spare)
    require_gpu
    # Kept as a negative/mechanistic control from the AntMaze experiment.
    run_hrm humanoidmaze_giant 0 mixer 1 2 2 True 256 hm_h2l2_full_m256_s0
    ;;
  *)
    echo "Unknown queue: ${QUEUE}" >&2
    print_plan >&2
    exit 2
    ;;
esac
