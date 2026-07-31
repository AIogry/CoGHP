#!/usr/bin/env bash
set -euo pipefail

# Run two identical CoGHP AntMaze-Giant experiments on separate GPUs and
# compare both checkpoints plus all non-wall-clock CSV metrics.
#
# Usage:
#   bash scripts/test_reproducibility.sh
#
# Optional overrides:
#   SEED=0 GPU_A=0 GPU_B=1 WANDB_MODE=offline \
#     bash scripts/test_reproducibility.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMPLS_DIR="${PROJECT_ROOT}/impls"
PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/exp/test}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

SEED="${SEED:-0}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-CoGHP_Reproducibility}"
TEST_ID="${TEST_ID:-$(date +%Y%m%d_%H%M%S)}"

ENV_NAME="antmaze-medium-navigate-v0"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_SUBGOALS="${NUM_SUBGOALS:-1}"
SUBGOAL_STEPS="${SUBGOAL_STEPS:-25}"
FEATURE_DIM="${FEATURE_DIM:-128}"

GROUP_A="coghp_baseline_repro_${TEST_ID}_replica_a"
GROUP_B="coghp_baseline_repro_${TEST_ID}_replica_b"
LOG_A="${LOG_DIR}/${GROUP_A}.log"
LOG_B="${LOG_DIR}/${GROUP_B}.log"

export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONHASHSEED="${SEED}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_deterministic_ops"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops"
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable is missing: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${DATASET_DIR}/${ENV_NAME}.npz" ]]; then
  echo "Training dataset is missing: ${DATASET_DIR}/${ENV_NAME}.npz" >&2
  exit 2
fi
if [[ ! -f "${DATASET_DIR}/${ENV_NAME}-val.npz" ]]; then
  echo "Validation dataset is missing: ${DATASET_DIR}/${ENV_NAME}-val.npz" >&2
  exit 2
fi
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  echo "GPU_A and GPU_B must identify different GPUs." >&2
  exit 2
fi

COMMON_ARGS=(
  --agent=agents/coghp.py
  --env_name="${ENV_NAME}"
  --seed="${SEED}"
  --wandb_project="${WANDB_PROJECT}"
  --wandb_mode="${WANDB_MODE}"
  --save_dir="${OUTPUT_ROOT}"
  --train_steps="${TRAIN_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --eval_episodes="${EVAL_EPISODES}"
  --video_episodes=0
  --eval_on_cpu=1
  --fixed_validation_batch=True
  --require_deterministic_gpu=True
  --agent.batch_size="${BATCH_SIZE}"
  --agent.num_subgoals="${NUM_SUBGOALS}"
  --agent.subgoal_steps="${SUBGOAL_STEPS}"
  --agent.feature_dim="${FEATURE_DIM}"
  --agent.mixer_hidden=32
  --agent.num_mixer_blocks=1
  --agent.causal_mixer=True
  --agent.action_use_full_subgoal_chain=True
  --agent.share_mixer_weights=False
  --agent.separate_action_mixer=False
)

run_replica() {
  local gpu="$1"
  local group="$2"
  local log_file="$3"

  (
    cd "${IMPLS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON}" main.py \
      --run_group="${group}" \
      "${COMMON_ARGS[@]}"
  ) >"${log_file}" 2>&1
}

find_run_dir() {
  local group="$1"
  find "${OUTPUT_ROOT}" \
    -type d \
    -path "*/${group}/sd$(printf '%03d' "${SEED}")_*" \
    -print | sort | tail -1
}

cleanup() {
  if [[ -n "${PID_A:-}" ]]; then
    kill "${PID_A}" 2>/dev/null || true
  fi
  if [[ -n "${PID_B:-}" ]]; then
    kill "${PID_B}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "Starting reproducibility test ${TEST_ID}"
echo "Replica A: GPU ${GPU_A}, ${GROUP_A}"
echo "Replica B: GPU ${GPU_B}, ${GROUP_B}"
echo "Results root: ${OUTPUT_ROOT}"
echo "Logs:"
echo "  ${LOG_A}"
echo "  ${LOG_B}"

run_replica "${GPU_A}" "${GROUP_A}" "${LOG_A}" &
PID_A=$!
run_replica "${GPU_B}" "${GROUP_B}" "${LOG_B}" &
PID_B=$!

echo "Replica A PID: ${PID_A}"
echo "Replica B PID: ${PID_B}"

set +e
wait "${PID_A}"
STATUS_A=$?
wait "${PID_B}"
STATUS_B=$?
set -e
PID_A=""
PID_B=""

if [[ "${STATUS_A}" -ne 0 || "${STATUS_B}" -ne 0 ]]; then
  echo "At least one replica failed." >&2
  echo "Replica A exit status: ${STATUS_A}" >&2
  echo "Replica B exit status: ${STATUS_B}" >&2
  echo "Inspect ${LOG_A} and ${LOG_B}." >&2
  exit 1
fi

RUN_A="$(find_run_dir "${GROUP_A}")"
RUN_B="$(find_run_dir "${GROUP_B}")"
if [[ -z "${RUN_A}" || -z "${RUN_B}" ]]; then
  echo "Could not locate both completed run directories." >&2
  exit 1
fi

echo "Replica A result: ${RUN_A}"
echo "Replica B result: ${RUN_B}"

epoch="${SAVE_INTERVAL}"
while (( epoch <= TRAIN_STEPS )); do
  JAX_PLATFORMS=cpu \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_reproducibility.py" \
    "${RUN_A}" \
    "${RUN_B}" \
    --epoch="${epoch}"
  epoch=$((epoch + SAVE_INTERVAL))
done

echo "Both replicas completed and passed exact reproducibility checks."
