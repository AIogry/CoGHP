#!/usr/bin/env bash
set -euo pipefail

# Train HCoGHP with CoGHP outer autoregression and inner HRM updates.
#
# One task:
#   SEED=0 ANT_GPU=0 bash scripts/train_hcoghp.sh antmaze_large
#   SEED=0 POINT_GPU=0 bash scripts/train_hcoghp.sh pointmaze_giant
#   SEED=0 ANT_GPU=1 bash scripts/train_hcoghp.sh antmaze_giant
#   SEED=0 HUMANOID_GPU=0 bash scripts/train_hcoghp.sh humanoidmaze_giant
#
# All three Giant tasks for one seed on a two-GPU machine:
#   SEED=0 POINT_GPU=0 ANT_GPU=1 HUMANOID_GPU=0 \
#     bash scripts/train_hcoghp.sh giant
#
# New HRM initialization can be overridden explicitly:
#   HCOGHP_L_CYCLES=2 HCOGHP_INIT_STD=0.1 \
#   HCOGHP_TRAINABLE_INIT=False bash scripts/train_hcoghp.sh antmaze_large

TASK="${1:-giant}"
SEED="${SEED:-0}"
RUN_ID="sd$(printf '%03d' "${SEED}")_$(date +%Y%m%d_%H%M%S)"

L_CYCLES="${HCOGHP_L_CYCLES:-2}"
INIT_STD="${HCOGHP_INIT_STD:-0.1}"
TRAINABLE_INIT="${HCOGHP_TRAINABLE_INIT:-False}"
LOW_NUM_BLOCKS="${HCOGHP_LOW_NUM_BLOCKS:-1}"
HIGH_NUM_BLOCKS="${HCOGHP_HIGH_NUM_BLOCKS:-1}"
READOUT_HIDDEN_DIM="${HCOGHP_READOUT_HIDDEN_DIM:-0}"
GRADIENT_MODE="${HCOGHP_GRADIENT_MODE:-stage_truncated}"
ENABLE_DIAGNOSTICS="${HCOGHP_DIAGNOSTICS:-0}"
ENABLE_FREE_VALIDATION="${HCOGHP_FREE_VALIDATION:-0}"

POINT_GPU="${POINT_GPU:-0}"
ANT_GPU="${ANT_GPU:-1}"
HUMANOID_GPU="${HUMANOID_GPU:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/hcoghp}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
OUTPUT_ROOT="${HCOGHP_OUTPUT_ROOT:-${DATA_ROOT}/exp}"
WANDB_PROJECT="${HCOGHP_WANDB_PROJECT:-HCoGHP}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"

if [[ "${DETERMINISTIC:-1}" == "1" ]]; then
  if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_deterministic_ops"* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops"
  fi
fi

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --eval_episodes=50
  --video_episodes=0
  --seed="${SEED}"
  --agent=agents/hcoghp.py
  --save_dir="${OUTPUT_ROOT}"
  --wandb_project="${WANDB_PROJECT}"
  --wandb_mode="${WANDB_MODE}"
  --agent.hrm_l_cycles="${L_CYCLES}"
  --agent.hrm_init_std="${INIT_STD}"
  --agent.hrm_trainable_init="${TRAINABLE_INIT}"
  --agent.hrm_low_num_blocks="${LOW_NUM_BLOCKS}"
  --agent.hrm_high_num_blocks="${HIGH_NUM_BLOCKS}"
  --agent.subgoal_readout_hidden_dim="${READOUT_HIDDEN_DIM}"
  --agent.hrm_gradient_mode="${GRADIENT_MODE}"
)

if [[ "${ENABLE_DIAGNOSTICS}" == "1" ]]; then
  COMMON_ARGS+=(--agent.enable_hcoghp_diagnostics=True)
fi

if [[ "${ENABLE_FREE_VALIDATION}" == "1" ]]; then
  COMMON_ARGS+=(--agent.enable_free_running_validation=True)
fi

# task_key|env_name|num_subgoals|subgoal_steps|feature_dim|extra_agent_args
TASK_SPECS=(
  "antmaze_large|antmaze-large-navigate-v0|1|50|128|"
  "pointmaze_giant|pointmaze-giant-navigate-v0|3|50|32|--agent.lr=1e-5"
  "antmaze_giant|antmaze-giant-navigate-v0|2|50|128|"
  "humanoidmaze_giant|humanoidmaze-giant-navigate-v0|2|100|128|--agent.discount=0.995"
)

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
train_path = os.path.join(dataset_dir, '${env_name}.npz')
val_path = os.path.join(dataset_dir, '${env_name}-val.npz')

print('Python:', sys.executable)
print('ogbench:', ogbench_file)
print('Dataset dir:', dataset_dir)
print('Train dataset exists:', os.path.exists(train_path), train_path)
print('Val dataset exists:', os.path.exists(val_path), val_path)

if not ogbench_file.startswith(project_root):
    raise SystemExit('ERROR: imported ogbench is not the local project copy')
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise SystemExit('ERROR: expected dataset files are missing')
"
  )
}

run_task() {
  local task_key="$1"
  local gpu_id="$2"
  local spec env_name num_subgoals subgoal_steps feature_dim extra_args run_group

  spec="$(task_spec_for "${task_key}")" || {
    echo "Unknown task: ${task_key}" >&2
    return 2
  }

  IFS='|' read -r task_key env_name num_subgoals subgoal_steps feature_dim extra_args <<< "${spec}"
  check_dataset_runtime "${env_name}"
  run_group="${task_key}_hcoghp_l${L_CYCLES}_init${INIT_STD}"

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

run_all_giant() {
  local point_log ant_log humanoid_log worker0_status worker1_status

  point_log="${LOG_DIR}/pointmaze_giant_l${L_CYCLES}_${RUN_ID}.log"
  ant_log="${LOG_DIR}/antmaze_giant_l${L_CYCLES}_${RUN_ID}.log"
  humanoid_log="${LOG_DIR}/humanoidmaze_giant_l${L_CYCLES}_${RUN_ID}.log"

  echo "Seed ${SEED}: PointMaze Giant on GPU ${POINT_GPU}, then HumanoidMaze Giant on GPU ${HUMANOID_GPU}."
  echo "Seed ${SEED}: AntMaze Giant on GPU ${ANT_GPU}."
  echo "Results root: ${OUTPUT_ROOT}/${WANDB_PROJECT}"

  (
    run_task pointmaze_giant "${POINT_GPU}" > "${point_log}" 2>&1
    run_task humanoidmaze_giant "${HUMANOID_GPU}" > "${humanoid_log}" 2>&1
  ) &
  WORKER0_PID=$!

  run_task antmaze_giant "${ANT_GPU}" > "${ant_log}" 2>&1 &
  WORKER1_PID=$!

  echo "Worker Point→Humanoid PID: ${WORKER0_PID}"
  echo "Worker Ant PID: ${WORKER1_PID}"
  echo "Logs:"
  echo "  ${point_log}"
  echo "  ${ant_log}"
  echo "  ${humanoid_log}"

  set +e
  wait "${WORKER0_PID}"
  worker0_status=$?
  wait "${WORKER1_PID}"
  worker1_status=$?
  set -e

  if [[ "${worker0_status}" -ne 0 || "${worker1_status}" -ne 0 ]]; then
    echo "At least one HCoGHP Giant task failed." >&2
    echo "Point→Humanoid worker exit code: ${worker0_status}" >&2
    echo "Ant worker exit code: ${worker1_status}" >&2
    return 1
  fi

  echo "All HCoGHP Giant tasks for seed ${SEED} completed successfully."
}

case "${TASK}" in
  large|antmaze_large)
    run_task antmaze_large "${ANT_GPU}"
    ;;
  pointmaze_giant)
    run_task pointmaze_giant "${POINT_GPU}"
    ;;
  antmaze_giant)
    run_task antmaze_giant "${ANT_GPU}"
    ;;
  humanoidmaze_giant)
    run_task humanoidmaze_giant "${HUMANOID_GPU}"
    ;;
  giant|all)
    run_all_giant
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Expected one of: large, antmaze_large, pointmaze_giant, antmaze_giant, humanoidmaze_giant, giant, all" >&2
    exit 2
    ;;
esac
