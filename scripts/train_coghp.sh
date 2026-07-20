#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_coghp.sh large
#   bash scripts/train_coghp.sh giant
#   bash scripts/train_coghp.sh all

TASK="${1:-all}"
RUN_ID="sd000_$(date +%Y%m%d_%H%M%S)"
LARGE_GPU="${LARGE_GPU:-0}"
GIANT_GPU="${GIANT_GPU:-1}"
COGHP_CAUSAL_MIXER="${COGHP_CAUSAL_MIXER:-True}"
COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN="${COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN:-True}"
COGHP_SHARE_MIXER_WEIGHTS="${COGHP_SHARE_MIXER_WEIGHTS:-False}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

if [[ "${TASK}" == *_no_causal ]]; then
  COGHP_CAUSAL_MIXER=False
  RUN_SUFFIX="${RUN_SUFFIX:-_no_causal}"
fi

if [[ "${TASK}" == *_last_subgoal_action ]]; then
  COGHP_ACTION_USE_FULL_SUBGOAL_CHAIN=False
  RUN_SUFFIX="${RUN_SUFFIX:-_last_subgoal_action}"
fi

if [[ "${TASK}" == *_shared_mixer ]]; then
  COGHP_SHARE_MIXER_WEIGHTS=True
  RUN_SUFFIX="${RUN_SUFFIX:-_shared_mixer}"
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
  local gpu_id="${1:-0}"
  local run_suffix="${2:-${RUN_SUFFIX}}"
  check_dataset_runtime "antmaze-large-navigate-v0"

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="antmaze_large_coghp${run_suffix}" \
      --env_name=antmaze-large-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=1 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

run_antmaze_giant() {
  local gpu_id="${1:-1}"
  local run_suffix="${2:-${RUN_SUFFIX}}"
  check_dataset_runtime "antmaze-giant-navigate-v0"

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="antmaze_giant_coghp${run_suffix}" \
      --env_name=antmaze-giant-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=2 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

case "${TASK}" in
  large)
    run_antmaze_large "${LARGE_GPU}"
    ;;

  giant)
    run_antmaze_giant "${GIANT_GPU}"
    ;;

  large_no_causal)
    run_antmaze_large "${LARGE_GPU}" "_no_causal"
    ;;

  giant_no_causal)
    run_antmaze_giant "${GIANT_GPU}" "_no_causal"
    ;;

  large_last_subgoal_action)
    run_antmaze_large "${LARGE_GPU}" "_last_subgoal_action"
    ;;

  giant_last_subgoal_action)
    run_antmaze_giant "${GIANT_GPU}" "_last_subgoal_action"
    ;;

  large_shared_mixer)
    run_antmaze_large "${LARGE_GPU}" "_shared_mixer"
    ;;

  giant_shared_mixer)
    run_antmaze_giant "${GIANT_GPU}" "_shared_mixer"
    ;;

  all)
    echo "Starting AntMaze Large on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" \
      > "${LOG_DIR}/antmaze_large_${RUN_ID}.log" 2>&1 &
    LARGE_PID=$!

    echo "Starting AntMaze Giant on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" \
      > "${LOG_DIR}/antmaze_giant_${RUN_ID}.log" 2>&1 &
    GIANT_PID=$!

    echo "Large PID: ${LARGE_PID}"
    echo "Giant PID: ${GIANT_PID}"
    echo "Logs:"
    echo "  ${LOG_DIR}/antmaze_large_${RUN_ID}.log"
    echo "  ${LOG_DIR}/antmaze_giant_${RUN_ID}.log"

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

    echo "Both training runs completed successfully."
    ;;

  all_no_causal)
    echo "Starting AntMaze Large without causal mixer on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" "_no_causal" \
      > "${LOG_DIR}/antmaze_large_no_causal_${RUN_ID}.log" 2>&1 &
    LARGE_PID=$!

    echo "Starting AntMaze Giant without causal mixer on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" "_no_causal" \
      > "${LOG_DIR}/antmaze_giant_no_causal_${RUN_ID}.log" 2>&1 &
    GIANT_PID=$!

    echo "Large PID: ${LARGE_PID}"
    echo "Giant PID: ${GIANT_PID}"
    echo "Logs:"
    echo "  ${LOG_DIR}/antmaze_large_no_causal_${RUN_ID}.log"
    echo "  ${LOG_DIR}/antmaze_giant_no_causal_${RUN_ID}.log"

    set +e
    wait "${LARGE_PID}"
    LARGE_STATUS=$?

    wait "${GIANT_PID}"
    GIANT_STATUS=$?
    set -e

    if [[ "${LARGE_STATUS}" -ne 0 || "${GIANT_STATUS}" -ne 0 ]]; then
      echo "At least one no-causal training process failed." >&2
      echo "Large exit code: ${LARGE_STATUS}" >&2
      echo "Giant exit code: ${GIANT_STATUS}" >&2
      exit 1
    fi

    echo "Both no-causal training runs completed successfully."
    ;;

  all_last_subgoal_action)
    echo "Starting AntMaze Large with last-subgoal-only action context on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" "_last_subgoal_action" \
      > "${LOG_DIR}/antmaze_large_last_subgoal_action_${RUN_ID}.log" 2>&1 &
    LARGE_PID=$!

    echo "Starting AntMaze Giant with last-subgoal-only action context on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" "_last_subgoal_action" \
      > "${LOG_DIR}/antmaze_giant_last_subgoal_action_${RUN_ID}.log" 2>&1 &
    GIANT_PID=$!

    echo "Large PID: ${LARGE_PID}"
    echo "Giant PID: ${GIANT_PID}"
    echo "Logs:"
    echo "  ${LOG_DIR}/antmaze_large_last_subgoal_action_${RUN_ID}.log"
    echo "  ${LOG_DIR}/antmaze_giant_last_subgoal_action_${RUN_ID}.log"

    set +e
    wait "${LARGE_PID}"
    LARGE_STATUS=$?

    wait "${GIANT_PID}"
    GIANT_STATUS=$?
    set -e

    if [[ "${LARGE_STATUS}" -ne 0 || "${GIANT_STATUS}" -ne 0 ]]; then
      echo "At least one last-subgoal-action training process failed." >&2
      echo "Large exit code: ${LARGE_STATUS}" >&2
      echo "Giant exit code: ${GIANT_STATUS}" >&2
      exit 1
    fi

    echo "Both last-subgoal-action training runs completed successfully."
    ;;

  all_shared_mixer)
    echo "Starting AntMaze Large with shared mixer weights on GPU ${LARGE_GPU}..."
    run_antmaze_large "${LARGE_GPU}" "_shared_mixer" \
      > "${LOG_DIR}/antmaze_large_shared_mixer_${RUN_ID}.log" 2>&1 &
    LARGE_PID=$!

    echo "Starting AntMaze Giant with shared mixer weights on GPU ${GIANT_GPU}..."
    run_antmaze_giant "${GIANT_GPU}" "_shared_mixer" \
      > "${LOG_DIR}/antmaze_giant_shared_mixer_${RUN_ID}.log" 2>&1 &
    GIANT_PID=$!

    echo "Large PID: ${LARGE_PID}"
    echo "Giant PID: ${GIANT_PID}"
    echo "Logs:"
    echo "  ${LOG_DIR}/antmaze_large_shared_mixer_${RUN_ID}.log"
    echo "  ${LOG_DIR}/antmaze_giant_shared_mixer_${RUN_ID}.log"

    set +e
    wait "${LARGE_PID}"
    LARGE_STATUS=$?

    wait "${GIANT_PID}"
    GIANT_STATUS=$?
    set -e

    if [[ "${LARGE_STATUS}" -ne 0 || "${GIANT_STATUS}" -ne 0 ]]; then
      echo "At least one shared-mixer training process failed." >&2
      echo "Large exit code: ${LARGE_STATUS}" >&2
      echo "Giant exit code: ${GIANT_STATUS}" >&2
      exit 1
    fi

    echo "Both shared-mixer training runs completed successfully."
    ;;

  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Expected one of: large, giant, all, large_no_causal, giant_no_causal, all_no_causal, large_last_subgoal_action, giant_last_subgoal_action, all_last_subgoal_action, large_shared_mixer, giant_shared_mixer, all_shared_mixer" >&2
    exit 2
    ;;
esac
