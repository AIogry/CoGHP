#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_hcoghp.sh large
#   bash scripts/train_hcoghp.sh giant
#   bash scripts/train_hcoghp.sh all
#
# Optional:
#   HCOGHP_L_CYCLES=4 bash scripts/train_hcoghp.sh all

TASK="${1:-all}"
RUN_ID="sd000_$(date +%Y%m%d_%H%M%S)"
L_CYCLES="${HCOGHP_L_CYCLES:-2}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/data/qijunrong/06-RL/offline-rl"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${DATA_ROOT}/logs/hcoghp"
DATASET_DIR="${DATA_ROOT}/data/raw_ogbench"
export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --eval_episodes=50
  --video_episodes=0
  --agent=agents/hcoghp.py
  --save_dir="${DATA_ROOT}/exp"
  --agent.hrm_l_cycles="${L_CYCLES}"
)

check_dataset_runtime() {
  local env_name="$1"

  (
    cd "${IMPLS_DIR}"
    python -c "
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
  local gpu_id="${1:-0}"
  check_dataset_runtime "antmaze-large-navigate-v0"

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python main.py \
      --run_group="antmaze_large_hcoghp_l${L_CYCLES}" \
      --env_name=antmaze-large-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=1 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

run_antmaze_giant() {
  local gpu_id="${1:-1}"
  check_dataset_runtime "antmaze-giant-navigate-v0"

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python main.py \
      --run_group="antmaze_giant_hcoghp_l${L_CYCLES}" \
      --env_name=antmaze-giant-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=2 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

case "${TASK}" in
  large)
    run_antmaze_large 0
    ;;

  giant)
    run_antmaze_giant 1
    ;;

  all)
    echo "Starting HCoGHP AntMaze Large on GPU 0 with L_CYCLES=${L_CYCLES}..."
    run_antmaze_large 0 \
      > "${LOG_DIR}/antmaze_large_l${L_CYCLES}_${RUN_ID}.log" 2>&1 &
    LARGE_PID=$!

    echo "Starting HCoGHP AntMaze Giant on GPU 1 with L_CYCLES=${L_CYCLES}..."
    run_antmaze_giant 1 \
      > "${LOG_DIR}/antmaze_giant_l${L_CYCLES}_${RUN_ID}.log" 2>&1 &
    GIANT_PID=$!

    echo "Large PID: ${LARGE_PID}"
    echo "Giant PID: ${GIANT_PID}"
    echo "Logs:"
    echo "  ${LOG_DIR}/antmaze_large_l${L_CYCLES}_${RUN_ID}.log"
    echo "  ${LOG_DIR}/antmaze_giant_l${L_CYCLES}_${RUN_ID}.log"

    set +e
    wait "${LARGE_PID}"
    LARGE_STATUS=$?

    wait "${GIANT_PID}"
    GIANT_STATUS=$?
    set -e

    if [[ "${LARGE_STATUS}" -ne 0 || "${GIANT_STATUS}" -ne 0 ]]; then
      echo "At least one training process failed." >&2
      echo "Large exit code: ${LARGE_STATUS}" >&2
      echo "Giant exit code: ${GIANT_STATUS}" >&2
      exit 1
    fi

    echo "Both HCoGHP training runs completed successfully."
    ;;

  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Expected one of: large, giant, all" >&2
    exit 2
    ;;
esac
