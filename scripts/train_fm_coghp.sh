#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_fm_coghp.sh large
#   bash scripts/train_fm_coghp.sh giant
#   bash scripts/train_fm_coghp.sh all
#
# Common overrides:
#   FLOW_SELECTOR=local_crl FLOW_STEPS=8 FLOW_CANDIDATES=4 \
#     bash scripts/train_fm_coghp.sh giant

TASK="${1:-all}"
SEED="${SEED:-0}"
RUN_ID="sd$(printf '%03d' "${SEED}")_$(date +%Y%m%d_%H%M%S)"
LARGE_GPU="${LARGE_GPU:-0}"
GIANT_GPU="${GIANT_GPU:-1}"
FLOW_SELECTOR="${FLOW_SELECTOR:-none}"
FLOW_STEPS="${FLOW_STEPS:-8}"
FLOW_CANDIDATES="${FLOW_CANDIDATES:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
EVAL_TASKS="${EVAL_TASKS:-}"
ENABLE_FM_DIAGNOSTICS="${ENABLE_FM_DIAGNOSTICS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/fm_coghp}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --seed="${SEED}"
  --train_steps="${TRAIN_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --eval_episodes="${EVAL_EPISODES}"
  --video_episodes="${VIDEO_EPISODES}"
  --eval_on_cpu="${EVAL_ON_CPU}"
  --wandb_project=FM_CoGHP
  --wandb_mode="${WANDB_MODE}"
  --agent=agents/fm_coghp.py
  --save_dir="${DATA_ROOT}/exp"
  --agent.flow_steps="${FLOW_STEPS}"
  --agent.flow_num_candidates="${FLOW_CANDIDATES}"
  --agent.flow_selector="${FLOW_SELECTOR}"
  --agent.flow_num_blocks=2
  --agent.flow_loss_weight=1.0
  --agent.flow_clean_loss_weight=0.1
  --agent.flow_high_aux_weight=0.0
  --agent.crl_loss_weight=1.0
)

if [[ -n "${EVAL_TASKS}" ]]; then
  COMMON_ARGS+=(--eval_tasks="${EVAL_TASKS}")
fi

if [[ "${ENABLE_FM_DIAGNOSTICS}" == "1" ]]; then
  COMMON_ARGS+=(--agent.enable_fm_diagnostics=True)
else
  COMMON_ARGS+=(--agent.enable_fm_diagnostics=False)
fi

check_dataset_runtime() {
  local env_name="$1"
  local train_path="${DATASET_DIR}/${env_name}.npz"
  local val_path="${DATASET_DIR}/${env_name}-val.npz"

  if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable is unavailable: ${PYTHON}" >&2
    return 1
  fi
  if [[ ! -f "${train_path}" || ! -f "${val_path}" ]]; then
    echo "Missing dataset files: ${train_path} or ${val_path}" >&2
    return 1
  fi

  (
    cd "${IMPLS_DIR}"
    "${PYTHON}" -c "
import os
import ogbench
project_root = os.path.abspath('${PROJECT_ROOT}')
ogbench_file = os.path.abspath(ogbench.__file__)
print('Python:', os.path.realpath('${PYTHON}'))
print('ogbench:', ogbench_file)
if not ogbench_file.startswith(project_root):
    raise SystemExit('ERROR: imported ogbench is not the local project copy')
"
  )
}

run_large() {
  local gpu_id="$1"
  check_dataset_runtime antmaze-large-navigate-v0
  (
    cd "${IMPLS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" main.py \
      --run_group="antmaze_large_fm_coghp_${FLOW_SELECTOR}" \
      --env_name=antmaze-large-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=1 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

run_giant() {
  local gpu_id="$1"
  check_dataset_runtime antmaze-giant-navigate-v0
  (
    cd "${IMPLS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" main.py \
      --run_group="antmaze_giant_fm_coghp_${FLOW_SELECTOR}" \
      --env_name=antmaze-giant-navigate-v0 \
      "${COMMON_ARGS[@]}" \
      --agent.num_subgoals=2 \
      --agent.subgoal_steps=50 \
      --agent.feature_dim=128
  )
}

wait_pair() {
  local left_pid="$1"
  local right_pid="$2"
  local left_status right_status
  set +e
  wait "${left_pid}"
  left_status=$?
  wait "${right_pid}"
  right_status=$?
  set -e
  if [[ "${left_status}" -ne 0 || "${right_status}" -ne 0 ]]; then
    echo "FM-CoGHP training failed: large=${left_status}, giant=${right_status}" >&2
    return 1
  fi
}

case "${TASK}" in
  large)
    run_large "${LARGE_GPU}"
    ;;
  giant)
    run_giant "${GIANT_GPU}"
    ;;
  all)
    large_log="${LOG_DIR}/antmaze_large_${FLOW_SELECTOR}_${RUN_ID}.log"
    giant_log="${LOG_DIR}/antmaze_giant_${FLOW_SELECTOR}_${RUN_ID}.log"
    run_large "${LARGE_GPU}" > "${large_log}" 2>&1 &
    large_pid=$!
    run_giant "${GIANT_GPU}" > "${giant_log}" 2>&1 &
    giant_pid=$!
    echo "Large PID ${large_pid}: ${large_log}"
    echo "Giant PID ${giant_pid}: ${giant_log}"
    wait_pair "${large_pid}" "${giant_pid}"
    ;;
  *)
    echo "Unknown task: ${TASK}; expected large, giant, or all." >&2
    exit 2
    ;;
esac
