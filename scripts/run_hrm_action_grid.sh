#!/usr/bin/env bash
set -euo pipefail

# Run/print the clean split-action baseline and a compact HRM-Action pilot grid.
# No original shared planner/action CoGHP run is included.

TASK_FILTER="${1:-${TASK_FILTER:-all}}"
DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${TASK_FILTER}" in
  all) TASKS=(antmaze_giant humanoidmaze_giant) ;;
  antmaze_giant|antmaze-giant) TASKS=(antmaze_giant) ;;
  humanoidmaze_giant|humanoidmaze-giant) TASKS=(humanoidmaze_giant) ;;
  *) echo "Unknown task filter: ${TASK_FILTER}" >&2; exit 2 ;;
esac
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS must contain at least one comma-separated GPU ID." >&2
  exit 2
fi

PROFILES=(
  "mixer|1|1|1"
  "mixer|1|2|2"
  "attention|1|2|2"
)
run_count=0

run_or_print() {
  local -a command=("$@")
  run_count=$((run_count + 1))
  printf 'RUN %02d:' "${run_count}"
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    (cd "${PROJECT_ROOT}" && "${command[@]}")
  fi
}

next_gpu() {
  printf '%s\n' "${GPU_ARRAY[$((run_count % ${#GPU_ARRAY[@]}))]}"
}

for task in "${TASKS[@]}"; do
  gpu="$(next_gpu)"
  if [[ "${task}" == "antmaze_giant" ]]; then
    baseline_gpu_var="GIANT_GPU=${gpu}"
  else
    baseline_gpu_var="LARGE_GPU=${gpu}"
  fi
  run_or_print env \
    "SEED=${SEED}" \
    "${baseline_gpu_var}" \
    "WANDB_MODE=${WANDB_MODE}" \
    "DATA_ROOT=${DATA_ROOT}" \
    "PYTHON=${PYTHON}" \
    bash scripts/train_coghp.sh "${task}_split_action_baseline"

  for profile in "${PROFILES[@]}"; do
    IFS='|' read -r backbone num_layers h_cycles l_cycles <<< "${profile}"
    gpu="$(next_gpu)"
    run_or_print env \
      "SEED=${SEED}" \
      "GPU_ID=${gpu}" \
      "WANDB_MODE=${WANDB_MODE}" \
      "DATA_ROOT=${DATA_ROOT}" \
      "PYTHON=${PYTHON}" \
      "ACTION_HRM_BACKBONE=${backbone}" \
      "ACTION_HRM_NUM_LAYERS=${num_layers}" \
      "ACTION_HRM_H_CYCLES=${h_cycles}" \
      "ACTION_HRM_L_CYCLES=${l_cycles}" \
      ACTION_HRM_BPTT=True \
      ACTION_CHAIN_SOURCE=teacher \
      bash scripts/train_hrm_action_coghp.sh "${task}"
  done
done

echo "SUMMARY total=${run_count} tasks=${#TASKS[@]} seed=${SEED} dry_run=${DRY_RUN}"
