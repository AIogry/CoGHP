#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_hcoghp_ss.sh large_ss25
#   bash scripts/train_hcoghp_ss.sh large_ss50
#   bash scripts/train_hcoghp_ss.sh giant_ss25
#   bash scripts/train_hcoghp_ss.sh giant_ss50
#   bash scripts/train_hcoghp_ss.sh all

TASK="${1:-all}"
RUN_ID="sd000_$(date +%Y%m%d_%H%M%S)"
LARGE_GPU="${LARGE_GPU:-0}"
GIANT_GPU="${GIANT_GPU:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/hcoghp_ss}"
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
  --agent=agents/hcoghp.py
  --save_dir="${DATA_ROOT}/exp"
  --agent.hrm_l_cycles=2
  --agent.scheduled_sampling_enabled=True
  --agent.scheduled_sampling_start_step=100000
  --agent.scheduled_sampling_end_step=300000
  --agent.scheduled_sampling_use_mode=True
  --agent.scheduled_sampling_stop_gradient=True
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

run_antmaze_large() {
  local gpu_id="$1"
  local ss_label="$2"
  local ss_prob="$3"
  check_dataset_runtime "antmaze-large-navigate-v0"

  (
    cd "${IMPLS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="antmaze_large_hcoghp_${ss_label}" \
      --env_name=antmaze-large-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.scheduled_sampling_max_prob="${ss_prob}" \
      --agent.num_subgoals=1 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

run_antmaze_giant() {
  local gpu_id="$1"
  local ss_label="$2"
  local ss_prob="$3"
  check_dataset_runtime "antmaze-giant-navigate-v0"

  (
    cd "${IMPLS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="antmaze_giant_hcoghp_${ss_label}" \
      --env_name=antmaze-giant-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.scheduled_sampling_max_prob="${ss_prob}" \
      --agent.num_subgoals=2 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

wait_pair() {
  local left_pid="$1"
  local right_pid="$2"
  set +e
  wait "${left_pid}"
  local left_status=$?
  wait "${right_pid}"
  local right_status=$?
  set -e
  if [[ "${left_status}" -ne 0 || "${right_status}" -ne 0 ]]; then
    echo "At least one training process failed." >&2
    echo "Left exit code: ${left_status}" >&2
    echo "Right exit code: ${right_status}" >&2
    exit 1
  fi
}

run_all_for_prob() {
  local ss_label="$1"
  local ss_prob="$2"

  echo "Starting AntMaze Large ${ss_label} on GPU ${LARGE_GPU}..."
  run_antmaze_large "${LARGE_GPU}" "${ss_label}" "${ss_prob}" \
    > "${LOG_DIR}/antmaze_large_${ss_label}_${RUN_ID}.log" 2>&1 &
  local large_pid=$!

  echo "Starting AntMaze Giant ${ss_label} on GPU ${GIANT_GPU}..."
  run_antmaze_giant "${GIANT_GPU}" "${ss_label}" "${ss_prob}" \
    > "${LOG_DIR}/antmaze_giant_${ss_label}_${RUN_ID}.log" 2>&1 &
  local giant_pid=$!

  echo "Large PID: ${large_pid}"
  echo "Giant PID: ${giant_pid}"
  echo "Logs:"
  echo "  ${LOG_DIR}/antmaze_large_${ss_label}_${RUN_ID}.log"
  echo "  ${LOG_DIR}/antmaze_giant_${ss_label}_${RUN_ID}.log"

  wait_pair "${large_pid}" "${giant_pid}"
}

case "${TASK}" in
  large_ss25)
    echo "Starting AntMaze Large ss25 on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" ss25 0.25 \
      > "${LOG_DIR}/antmaze_large_ss25_${RUN_ID}.log" 2>&1
    echo "Log: ${LOG_DIR}/antmaze_large_ss25_${RUN_ID}.log"
    ;;
  large_ss50)
    echo "Starting AntMaze Large ss50 on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" ss50 0.5 \
      > "${LOG_DIR}/antmaze_large_ss50_${RUN_ID}.log" 2>&1
    echo "Log: ${LOG_DIR}/antmaze_large_ss50_${RUN_ID}.log"
    ;;
  giant_ss25)
    echo "Starting AntMaze Giant ss25 on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" ss25 0.25 \
      > "${LOG_DIR}/antmaze_giant_ss25_${RUN_ID}.log" 2>&1
    echo "Log: ${LOG_DIR}/antmaze_giant_ss25_${RUN_ID}.log"
    ;;
  giant_ss50)
    echo "Starting AntMaze Giant ss50 on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" ss50 0.5 \
      > "${LOG_DIR}/antmaze_giant_ss50_${RUN_ID}.log" 2>&1
    echo "Log: ${LOG_DIR}/antmaze_giant_ss50_${RUN_ID}.log"
    ;;
  all)
    run_all_for_prob ss25 0.25
    run_all_for_prob ss50 0.5
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Expected one of: large_ss25, large_ss50, giant_ss25, giant_ss50, all" >&2
    exit 2
    ;;
esac
