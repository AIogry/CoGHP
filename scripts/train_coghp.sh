#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_coghp.sh antmaze_large
#   bash scripts/train_coghp.sh pointmaze_giant
#   bash scripts/train_coghp.sh humanoidmaze_large
#   bash scripts/train_coghp.sh all
#
# Backward-compatible aliases:
#   bash scripts/train_coghp.sh large   # antmaze_large
#   bash scripts/train_coghp.sh giant   # antmaze_giant
#
# Optional:
#   GPUS="0 1 2 3" MAX_PARALLEL=4 bash scripts/train_coghp.sh all
#   COGHP_CAUSAL_MIXER=False bash scripts/train_coghp.sh antmaze_large

TASK="${1:-all}"
RUN_ID="sd000_$(date +%Y%m%d_%H%M%S)"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0 1}}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LARGE_GPU="${LARGE_GPU:-0}"
GIANT_GPU="${GIANT_GPU:-1}"
COGHP_CAUSAL_MIXER="${COGHP_CAUSAL_MIXER:-True}"
COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN="${COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN:-True}"
COGHP_SHARE_MIXER_WEIGHTS="${COGHP_SHARE_MIXER_WEIGHTS:-False}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

BASE_TASK="${TASK}"
if [[ "${TASK}" == *_no_causal ]]; then
  COGHP_CAUSAL_MIXER=False
  RUN_SUFFIX="${RUN_SUFFIX:-_no_causal}"
  BASE_TASK="${BASE_TASK%_no_causal}"
fi

if [[ "${TASK}" == *_last_subgoal_action ]]; then
  COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN=False
  RUN_SUFFIX="${RUN_SUFFIX:-_last_subgoal_action}"
  BASE_TASK="${BASE_TASK%_last_subgoal_action}"
fi

if [[ "${TASK}" == *_shared_mixer ]]; then
  COGHP_SHARE_MIXER_WEIGHTS=True
  RUN_SUFFIX="${RUN_SUFFIX:-_shared_mixer}"
  BASE_TASK="${BASE_TASK%_shared_mixer}"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/coghp}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --eval_episodes=50
  --video_episodes=0
  --agent=agents/coghp.py
  --save_dir="${DATA_ROOT}/exp"
  --agent.causal_mixer="${COGHP_CAUSAL_MIXER}"
  --agent.action_use_full_subgoal_chain="${COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN}"
  --agent.share_mixer_weights="${COGHP_SHARE_MIXER_WEIGHTS}"
)

# task_key|env_name|num_subgoals|subgoal_steps|feature_dim|extra_agent_args
TASK_SPECS=(
  "pointmaze_medium|pointmaze-medium-navigate-v0|1|25|32|"
  "pointmaze_large|pointmaze-large-navigate-v0|1|50|32|"
  "pointmaze_giant|pointmaze-giant-navigate-v0|3|50|32|--agent.lr=1e-5"
  "pointmaze_teleport|pointmaze-teleport-navigate-v0|1|50|32|"
  "pointmaze_teleport_stitch|pointmaze-teleport-stitch-v0|1|50|32|--agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5"
  "antmaze_medium|antmaze-medium-navigate-v0|1|25|128|"
  "antmaze_large|antmaze-large-navigate-v0|1|50|128|"
  "antmaze_giant|antmaze-giant-navigate-v0|2|50|128|"
  "antmaze_teleport|antmaze-teleport-navigate-v0|1|50|128|"
  "antmaze_teleport_stitch|antmaze-teleport-stitch-v0|1|50|128|--agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5"
  "antmaze_teleport_explore|antmaze-teleport-explore-v0|1|50|128|--agent.actor_p_trajgoal=0.0 --agent.actor_p_randomgoal=1.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0"
  "humanoidmaze_medium|humanoidmaze-medium-navigate-v0|1|100|128|--agent.discount=0.995"
  "humanoidmaze_large|humanoidmaze-large-navigate-v0|1|100|128|--agent.discount=0.995"
  "humanoidmaze_giant|humanoidmaze-giant-navigate-v0|2|100|128|--agent.discount=0.995"
  "humanoidmaze_teleport|humanoidmaze-teleport-navigate-v0|1|100|128|--agent.discount=0.995"
)

POINTMAZE_TASKS=(
  pointmaze_medium
  pointmaze_large
  pointmaze_giant
  pointmaze_teleport
  pointmaze_teleport_stitch
)

ANTMAZE_TASKS=(
  antmaze_medium
  antmaze_large
  antmaze_giant
  antmaze_teleport
  antmaze_teleport_stitch
  antmaze_teleport_explore
)

HUMANOIDMAZE_TASKS=(
  humanoidmaze_medium
  humanoidmaze_large
  humanoidmaze_giant
  humanoidmaze_teleport
)

ALL_TASKS=(
  "${POINTMAZE_TASKS[@]}"
  "${ANTMAZE_TASKS[@]}"
  "${HUMANOIDMAZE_TASKS[@]}"
)

check_dataset_runtime() {
  local env_name="$1"

  (
    cd "${IMPLS_DIR}"
    "${PYTHON}" -c "
import os
import sys
import ogbench
import ogbench.utils

project_root = os.path.abspath('${PROJECT_ROOT}')
dataset_dir = os.path.abspath('${DATASET_DIR}')
ogbench_file = os.path.abspath(ogbench.__file__)
utils_file = os.path.abspath(ogbench.utils.__file__)
train_path = os.path.join(dataset_dir, '${env_name}.npz')
val_path = os.path.join(dataset_dir, '${env_name}-val.npz')

print('Python:', sys.executable)
print('ogbench:', ogbench_file)
print('ogbench.utils:', utils_file)
print('DEFAULT_DATASET_DIR:', ogbench.utils.DEFAULT_DATASET_DIR)
print('Expected dataset dir:', dataset_dir)
print('Train dataset exists:', os.path.exists(train_path), train_path)
print('Val dataset exists:', os.path.exists(val_path), val_path)

if not ogbench_file.startswith(project_root):
    raise SystemExit('ERROR: imported ogbench is not the local project copy')
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise SystemExit('ERROR: expected dataset files are missing')
"
  )
}

task_spec_for() {
  local task_key="$1"
  local spec

  for spec in "${TASK_SPECS[@]}"; do
    if [[ "${spec%%|*}" == "${task_key}" ]]; then
      printf '%s\n' "${spec}"
      return 0
    fi
  done

  return 1
}

dataset_available_for_task() {
  local task_key="$1"
  local spec env_name _num_subgoals _subgoal_steps _feature_dim _extra_args

  task_key="$(normalize_task_key "${task_key}")"
  spec="$(task_spec_for "${task_key}")" || return 1
  IFS='|' read -r _task_key env_name _num_subgoals _subgoal_steps _feature_dim _extra_args <<< "${spec}"

  [[ -f "${DATASET_DIR}/${env_name}.npz" && -f "${DATASET_DIR}/${env_name}-val.npz" ]]
}

normalize_task_key() {
  local task_key="$1"

  case "${task_key}" in
    medium)
      printf '%s\n' "antmaze_medium"
      ;;
    large)
      printf '%s\n' "antmaze_large"
      ;;
    giant)
      printf '%s\n' "antmaze_giant"
      ;;
    teleport)
      printf '%s\n' "antmaze_teleport"
      ;;
    *)
      printf '%s\n' "${task_key}"
      ;;
  esac
}

run_task() {
  local task_key="$1"
  local gpu_id="$2"
  local run_suffix="${3:-${RUN_SUFFIX}}"
  local spec env_name num_subgoals subgoal_steps feature_dim extra_args run_group

  task_key="$(normalize_task_key "${task_key}")"
  spec="$(task_spec_for "${task_key}")" || {
    echo "Unknown task: ${task_key}" >&2
    return 2
  }

  IFS='|' read -r task_key env_name num_subgoals subgoal_steps feature_dim extra_args <<< "${spec}"
  check_dataset_runtime "${env_name}"
  run_group="${task_key}_coghp${run_suffix}"

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="${run_group}" \
      --env_name="${env_name}" \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals="${num_subgoals}" \
      --agent.subgoal_steps="${subgoal_steps}" \
      --agent.feature_dim="${feature_dim}" \
      ${extra_args}
  )
}

tasks_for_group() {
  local group="$1"

  case "${group}" in
    pointmaze)
      printf '%s\n' "${POINTMAZE_TASKS[@]}"
      ;;
    antmaze)
      printf '%s\n' "${ANTMAZE_TASKS[@]}"
      ;;
    humanoidmaze)
      printf '%s\n' "${HUMANOIDMAZE_TASKS[@]}"
      ;;
    all)
      printf '%s\n' "${ALL_TASKS[@]}"
      ;;
    *)
      return 1
      ;;
  esac
}

wait_for_slot() {
  local -n running_ref="$1"
  local -n failed_ref="$2"
  local status

  while (( running_ref >= MAX_PARALLEL )); do
    set +e
    wait -n
    status=$?
    set -e
    running_ref=$((running_ref - 1))
    if [[ "${status}" -ne 0 ]]; then
      failed_ref=1
    fi
  done
}

run_task_list() {
  local tasks=("$@")
  local gpu_list=(${GPUS})
  local task_key gpu_id log_file i status failed running pid

  if (( ${#gpu_list[@]} == 0 )); then
    echo "No GPUs configured. Set GPUS, e.g. GPUS=\"0 1\"." >&2
    exit 2
  fi

  if (( MAX_PARALLEL < 1 )); then
    echo "MAX_PARALLEL must be at least 1." >&2
    exit 2
  fi

  i=0
  failed=0
  running=0
  for task_key in "${tasks[@]}"; do
    if ! dataset_available_for_task "${task_key}"; then
      echo "Skipping ${task_key}: dataset files are missing under ${DATASET_DIR}."
      continue
    fi

    wait_for_slot running failed
    gpu_id="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    log_file="${LOG_DIR}/${task_key}${RUN_SUFFIX}_${RUN_ID}.log"
    echo "Starting ${task_key} on GPU ${gpu_id}..."
    run_task "${task_key}" "${gpu_id}" "${RUN_SUFFIX}" > "${log_file}" 2>&1 &
    pid=$!
    running=$((running + 1))
    echo "  PID: ${pid}"
    echo "  Log: ${log_file}"
    i=$((i + 1))
  done

  while (( running > 0 )); do
    set +e
    wait -n
    status=$?
    set -e
    running=$((running - 1))
    if [[ "${status}" -ne 0 ]]; then
      echo "A training process failed with exit code ${status}." >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
}

print_tasks() {
  echo "Available tasks:"
  printf '  %s\n' "${ALL_TASKS[@]}"
  echo
  echo "Groups:"
  echo "  pointmaze"
  echo "  antmaze"
  echo "  humanoidmaze"
  echo "  all"
  echo
  echo "Backward-compatible aliases:"
  echo "  medium -> antmaze_medium"
  echo "  large -> antmaze_large"
  echo "  giant -> antmaze_giant"
  echo "  teleport -> antmaze_teleport"
}

run_group_or_task() {
  local task_key="$1"
  local normalized tasks

  normalized="$(normalize_task_key "${task_key}")"
  if tasks="$(tasks_for_group "${normalized}")"; then
    mapfile -t task_array <<< "${tasks}"
    run_task_list "${task_array[@]}"
  elif task_spec_for "${normalized}" >/dev/null; then
    local gpu_id="${LARGE_GPU}"
    if [[ "${normalized}" == "antmaze_giant" ]]; then
      gpu_id="${GIANT_GPU}"
    fi
    run_task "${normalized}" "${gpu_id}" "${RUN_SUFFIX}"
  else
    echo "Unknown task: ${TASK}" >&2
    print_tasks >&2
    exit 2
  fi
}

case "${BASE_TASK}" in
  list|--list|-l)
    print_tasks
    ;;
  *)
    run_group_or_task "${BASE_TASK}"
    ;;
esac
