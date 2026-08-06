#!/usr/bin/env bash
set -euo pipefail

# Dedicated launcher for CoGHP-FF-SwiGLU-RMSNorm-204k only.
#
# Safe workflow:
#   GIT_COMMIT_OVERRIDE=<sha> BASELINE_SEED_POLICY=seed0_only \
#     bash scripts/run_coghp_ff_swiglu_rms_204k_a800.sh preflight
#   GIT_COMMIT_OVERRIDE=<sha> BASELINE_SEED_POLICY=seed0_only \
#     bash scripts/run_coghp_ff_swiglu_rms_204k_a800.sh launch
#
# BASELINE_SEED_POLICY is deliberately mandatory because the existing noon
# HRM seed sets differ by task:
#   seed0_only -> Ant {0}, Humanoid {0}
#   match_noon -> Ant {0,1,2}, Humanoid {0}
#   all_three -> Ant {0,1,2}, Humanoid {0,1,2}

MODE="${1:-plan}"
BASELINE_SEED_POLICY="${BASELINE_SEED_POLICY:-}"
GIT_COMMIT_OVERRIDE="${GIT_COMMIT_OVERRIDE:-}"
WANDB_MODE="${WANDB_MODE:-online}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-1024}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-10}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/env_xhumanoidserver.sh"

DATA_ROOT="${A800_DATA_ROOT:-/nix/data/offline-rl}"
DATASET_DIR="${A800_DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
PYTHON="${A800_PYTHON:-/root/Tools/miniforge3/envs/brain_nav/bin/python}"
WANDB_PROJECT="CoGHPFFSwiGLU204k"
OUTPUT_ROOT="${DATA_ROOT}/exp"
PROJECT_OUTPUT_ROOT="${OUTPUT_ROOT}/${WANDB_PROJECT}"
LOG_ROOT="${DATA_ROOT}/logs/coghp_ff_swiglu_rms_204k"
PREFLIGHT_ROOT="${LOG_ROOT}/preflight"
SUMMARY_CSV="${PROJECT_OUTPUT_ROOT}/coghp_ff_swiglu_rms_204k_summary.csv"

TRAIN_STEPS=1000000
LOG_INTERVAL=5000
EVAL_INTERVAL=100000
SAVE_INTERVAL=1000000
EVAL_EPISODES=50
BATCH_SIZE=256

SOURCE_FILES=(
  "${PROJECT_ROOT}/impls/main.py"
  "${PROJECT_ROOT}/impls/utils/flax_utils.py"
  "${PROJECT_ROOT}/impls/utils/hrm.py"
  "${PROJECT_ROOT}/impls/utils/hrm_action_coghp_network.py"
  "${PROJECT_ROOT}/impls/agents/hrm_action_coghp.py"
  "${PROJECT_ROOT}/impls/preflight_coghp_ff_swiglu_204k.py"
  "${PROJECT_ROOT}/impls/summarize_coghp_ff_swiglu_204k.py"
  "${PROJECT_ROOT}/scripts/train_hrm_action_coghp.sh"
  "${PROJECT_ROOT}/scripts/run_coghp_ff_swiglu_rms_204k_a800.sh"
)
CODE_SIGNATURE="$({ sha256sum "${SOURCE_FILES[@]}"; } | sha256sum | awk '{print $1}')"

export DATA_ROOT DATASET_DIR OGBENCH_DATASET_DIR="${DATASET_DIR}" PYTHON
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/impls:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

require_provenance() {
  if [[ -z "${GIT_COMMIT_OVERRIDE}" ]]; then
    echo "GIT_COMMIT_OVERRIDE is required because the synchronized server tree has no .git directory." >&2
    exit 2
  fi
}

set_seed_policy() {
  case "${BASELINE_SEED_POLICY}" in
    seed0_only)
      ANT_SEEDS=(0)
      HUMANOID_SEEDS=(0)
      ;;
    match_noon)
      ANT_SEEDS=(0 1 2)
      HUMANOID_SEEDS=(0)
      ;;
    all_three)
      ANT_SEEDS=(0 1 2)
      HUMANOID_SEEDS=(0 1 2)
      ;;
    *)
      echo "BASELINE_SEED_POLICY must be seed0_only, match_noon, or all_three." >&2
      exit 2
      ;;
  esac
}

experiment_name() {
  local task="$1"
  local seed="$2"
  printf 'coghp_ff_swiglu_rms_mi256_%s_seed%s\n' "${task}" "${seed}"
}

env_name() {
  case "$1" in
    antmaze_giant) printf '%s\n' antmaze-giant-navigate-v0 ;;
    humanoidmaze_giant) printf '%s\n' humanoidmaze-giant-navigate-v0 ;;
    *) return 1 ;;
  esac
}

free_gpu_ids() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F, \
      -v max_mem="${GPU_MAX_USED_MIB}" \
      -v max_util="${GPU_MAX_UTILIZATION}" \
      '$2 + 0 < max_mem && $3 + 0 < max_util {gsub(/ /, "", $1); print $1}'
}

contains_gpu() {
  local needle="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ "${candidate}" == "${needle}" ]] && return 0
  done
  return 1
}

select_two_gpus() {
  mapfile -t FREE_GPUS < <(free_gpu_ids)
  if (( ${#FREE_GPUS[@]} < 2 )); then
    echo "At least two idle GPUs are required; free set: ${FREE_GPUS[*]:-none}." >&2
    exit 2
  fi

  if [[ -n "${ANT_GPU_OVERRIDE:-}" ]]; then
    ANT_GPU="${ANT_GPU_OVERRIDE}"
  elif contains_gpu 0 "${FREE_GPUS[@]}"; then
    ANT_GPU=0
  else
    ANT_GPU="${FREE_GPUS[0]}"
  fi
  contains_gpu "${ANT_GPU}" "${FREE_GPUS[@]}" || {
    echo "Requested Ant GPU ${ANT_GPU} is not idle." >&2
    exit 2
  }

  if [[ -n "${HUMANOID_GPU_OVERRIDE:-}" ]]; then
    HUMANOID_GPU="${HUMANOID_GPU_OVERRIDE}"
  elif contains_gpu 1 "${FREE_GPUS[@]}" && [[ "${ANT_GPU}" != 1 ]]; then
    HUMANOID_GPU=1
  else
    HUMANOID_GPU=""
    local candidate
    for candidate in "${FREE_GPUS[@]}"; do
      if [[ "${candidate}" != "${ANT_GPU}" ]]; then
        HUMANOID_GPU="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${HUMANOID_GPU}" || "${ANT_GPU}" == "${HUMANOID_GPU}" ]]; then
    echo "Could not assign two distinct idle GPUs." >&2
    exit 2
  fi
  contains_gpu "${HUMANOID_GPU}" "${FREE_GPUS[@]}" || {
    echo "Requested Humanoid GPU ${HUMANOID_GPU} is not idle." >&2
    exit 2
  }
}

preflight_path() {
  local task="$1"
  local seed="$2"
  printf '%s/%s.json\n' "${PREFLIGHT_ROOT}" "$(experiment_name "${task}" "${seed}")"
}

run_preflight_one() {
  local task="$1"
  local seed="$2"
  local gpu="$3"
  local output
  output="$(preflight_path "${task}" "${seed}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    "${PROJECT_ROOT}/impls/preflight_coghp_ff_swiglu_204k.py" \
    --env-name="$(env_name "${task}")" \
    --seed="${seed}" \
    --git-commit="${GIT_COMMIT_OVERRIDE}" \
    --code-signature="${CODE_SIGNATURE}" \
    --output="${output}" \
    >"${output%.json}.log" 2>&1
  "${PYTHON}" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed"
assert report["action_core_trainable_params"] == 204288
assert report["B_L_params"] == 102144
assert report["B_H_params"] == 102144
assert report["total_block_calls"] == 2
assert report["has_zL"] is False and report["has_zH"] is False
assert report["effective_h_cycles"] == "N/A"
assert report["effective_l_cycles"] == "N/A"
assert report["effective_gradient_mode"] == "N/A"
' "${output}"
  echo "Preflight passed: ${output}"
}

verify_preflight_one() {
  local task="$1"
  local seed="$2"
  local output
  output="$(preflight_path "${task}" "${seed}")"
  if [[ ! -f "${output}" ]]; then
    # The architecture contract is seed-independent.  Reuse the complete
    # seed-0 preflight for later seeds; every formal process still performs
    # its own 204288-parameter assertion during Agent initialization and
    # writes a checkpoint signature containing its actual seed.
    output="$(preflight_path "${task}" 0)"
  fi
  "${PYTHON}" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed"
assert report["environment"] == sys.argv[2]
assert report["checkpoint_metadata"]["signature"]["git_commit"] == sys.argv[3]
assert report["checkpoint_metadata"]["signature"]["code_signature"] == sys.argv[4]
assert report["action_core_trainable_params"] == 204288
assert report["total_block_calls"] == 2
' "${output}" "$(env_name "${task}")" "${GIT_COMMIT_OVERRIDE}" "${CODE_SIGNATURE}"
}

assert_fresh_output() {
  local name="$1"
  local target="${PROJECT_OUTPUT_ROOT}/${name}"
  if [[ -e "${target}" ]]; then
    echo "Refusing to reuse existing experiment directory: ${target}" >&2
    exit 2
  fi
  for target in \
    "${LOG_ROOT}/${name}.log" \
    "${LOG_ROOT}/${name}_gpu.csv" \
    "${LOG_ROOT}/${name}_summary.log"; do
    if [[ -e "${target}" ]]; then
      echo "Refusing to reuse existing experiment artifact: ${target}" >&2
      exit 2
    fi
  done
}

write_run_contract() {
  local name="$1"
  local task="$2"
  local seed="$3"
  local gpu="$4"
  local target="${PROJECT_OUTPUT_ROOT}/${name}"
  mkdir -p "${target}"
  "${PYTHON}" -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "experiment_name": sys.argv[2],
    "environment": sys.argv[3],
    "seed": int(sys.argv[4]),
    "gpu": int(sys.argv[5]),
    "action_core_type": "matched_feedforward",
    "action_block_type": "swiglu_rmsnorm",
    "mixer_intermediate_size": 256,
    "num_layers": 1,
    "feature_dim": 128,
    "num_subgoals": 2,
    "outer_cycles": 1,
    "h_cycles": "N/A",
    "l_cycles": "N/A",
    "gradient_mode": "N/A",
    "resume": False,
    "restore_path": None,
    "git_commit": sys.argv[6],
    "code_signature": sys.argv[7],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
' "${target}/run_contract.json" "${name}" "$(env_name "${task}")" \
    "${seed}" "${gpu}" "${GIT_COMMIT_OVERRIDE}" "${CODE_SIGNATURE}"
}

monitor_gpu() {
  local gpu="$1"
  local watched_pid="$2"
  local output="$3"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    nvidia-smi --id="${gpu}" \
      --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >>"${output}" || true
    sleep 2
  done
  nvidia-smi --id="${gpu}" \
    --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits >>"${output}" || true
}

print_training_command() {
  local task="$1"
  local seed="$2"
  local gpu="$3"
  local name
  name="$(experiment_name "${task}" "${seed}")"
  local command=(
    env
    "DATA_ROOT=${DATA_ROOT}"
    "DATASET_DIR=${DATASET_DIR}"
    "PYTHON=${PYTHON}"
    "SEED=${seed}"
    "GPU_ID=${gpu}"
    "WANDB_MODE=${WANDB_MODE}"
    "HRM_ACTION_WANDB_PROJECT=${WANDB_PROJECT}"
    "HRM_ACTION_OUTPUT_ROOT=${OUTPUT_ROOT}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "TRAIN_STEPS=${TRAIN_STEPS}"
    "LOG_INTERVAL=${LOG_INTERVAL}"
    "EVAL_INTERVAL=${EVAL_INTERVAL}"
    "SAVE_INTERVAL=${SAVE_INTERVAL}"
    "EVAL_EPISODES=${EVAL_EPISODES}"
    EVAL_ON_CPU=1
    "RUN_GROUP_OVERRIDE=${name}"
    ACTION_HRM_BACKBONE=mixer
    ACTION_CORE_TYPE=matched_feedforward
    ACTION_BLOCK_TYPE=swiglu_rmsnorm
    HRM_BLOCK_TYPE=swiglu_rmsnorm
    ACTION_HRM_NUM_LAYERS=1
    ACTION_HRM_H_CYCLES=2
    ACTION_HRM_L_CYCLES=2
    ACTION_HRM_BPTT=False
    ACTION_HRM_MIXER_INTERMEDIATE_SIZE=256
    HRM_MIXER_INTERMEDIATE_SIZE=256
    ACTION_HRM_INTERMEDIATE_SIZE=512
    ACTION_HRM_HEAD_DIM=32
    ACTION_CHAIN_SOURCE=teacher
    EXPECTED_ACTION_CORE_PARAMS=204288
    CHECKPOINT_SIGNATURE_REQUIRED=True
    "EXPERIMENT_GIT_COMMIT=${GIT_COMMIT_OVERRIDE}"
    "EXPERIMENT_CODE_SIGNATURE=${CODE_SIGNATURE}"
    bash "${PROJECT_ROOT}/scripts/train_hrm_action_coghp.sh" "${task}"
  )
  printf 'COMMAND %s (GPU %s):\n  ' "${name}" "${gpu}"
  printf '%q ' "${command[@]}"
  printf '> %q 2>&1\n' "${LOG_ROOT}/${name}.log"
  printf 'OUTPUT: %s\n' "${PROJECT_OUTPUT_ROOT}/${name}"
}

run_one() {
  local task="$1"
  local seed="$2"
  local gpu="$3"
  local name
  name="$(experiment_name "${task}" "${seed}")"
  local log_file="${LOG_ROOT}/${name}.log"
  local monitor_file="${LOG_ROOT}/${name}_gpu.csv"

  write_run_contract "${name}" "${task}" "${seed}" "${gpu}"
  echo "[$(date '+%F %T')] START ${name} GPU=${gpu} log=${log_file}"
  env \
    DATA_ROOT="${DATA_ROOT}" \
    DATASET_DIR="${DATASET_DIR}" \
    PYTHON="${PYTHON}" \
    SEED="${seed}" \
    GPU_ID="${gpu}" \
    WANDB_MODE="${WANDB_MODE}" \
    HRM_ACTION_WANDB_PROJECT="${WANDB_PROJECT}" \
    HRM_ACTION_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    EVAL_ON_CPU=1 \
    RUN_GROUP_OVERRIDE="${name}" \
    ACTION_HRM_BACKBONE=mixer \
    ACTION_CORE_TYPE=matched_feedforward \
    ACTION_BLOCK_TYPE=swiglu_rmsnorm \
    HRM_BLOCK_TYPE=swiglu_rmsnorm \
    ACTION_HRM_NUM_LAYERS=1 \
    ACTION_HRM_H_CYCLES=2 \
    ACTION_HRM_L_CYCLES=2 \
    ACTION_HRM_BPTT=False \
    ACTION_HRM_MIXER_INTERMEDIATE_SIZE=256 \
    HRM_MIXER_INTERMEDIATE_SIZE=256 \
    ACTION_HRM_INTERMEDIATE_SIZE=512 \
    ACTION_HRM_HEAD_DIM=32 \
    ACTION_CHAIN_SOURCE=teacher \
    EXPECTED_ACTION_CORE_PARAMS=204288 \
    CHECKPOINT_SIGNATURE_REQUIRED=True \
    EXPERIMENT_GIT_COMMIT="${GIT_COMMIT_OVERRIDE}" \
    EXPERIMENT_CODE_SIGNATURE="${CODE_SIGNATURE}" \
    bash "${PROJECT_ROOT}/scripts/train_hrm_action_coghp.sh" "${task}" \
    >"${log_file}" 2>&1 &
  local train_pid=$!
  monitor_gpu "${gpu}" "${train_pid}" "${monitor_file}" &
  local monitor_pid=$!
  local status=0
  wait "${train_pid}" || status=$?
  wait "${monitor_pid}" || true
  if (( status != 0 )); then
    echo "[$(date '+%F %T')] FAIL ${name} status=${status} log=${log_file}" >&2
    return "${status}"
  fi
  echo "[$(date '+%F %T')] DONE ${name}"
}

run_task_queue() {
  local task="$1"
  local gpu="$2"
  shift 2
  local seed
  for seed in "$@"; do
    run_one "${task}" "${seed}" "${gpu}"
  done
}

find_run_dir() {
  local name="$1"
  find "${PROJECT_OUTPUT_ROOT}/${name}" \
    -mindepth 1 -maxdepth 1 -type d -name 'sd*' -print
}

summarize_all() {
  if [[ -e "${SUMMARY_CSV}" ]]; then
    echo "Refusing to overwrite existing summary: ${SUMMARY_CSV}" >&2
    return 1
  fi
  local summary_temp="${SUMMARY_CSV}.tmp.${BASHPID}"
  if [[ -e "${summary_temp}" ]]; then
    echo "Refusing to reuse temporary summary: ${summary_temp}" >&2
    return 1
  fi
  local task seed name run_dir monitor_file
  for task in antmaze_giant humanoidmaze_giant; do
    if [[ "${task}" == antmaze_giant ]]; then
      seeds=("${ANT_SEEDS[@]}")
    else
      seeds=("${HUMANOID_SEEDS[@]}")
    fi
    for seed in "${seeds[@]}"; do
      name="$(experiment_name "${task}" "${seed}")"
      mapfile -t run_dirs < <(find_run_dir "${name}")
      if (( ${#run_dirs[@]} != 1 )); then
        echo "Expected one run directory for ${name}, found ${#run_dirs[@]}." >&2
        return 1
      fi
      run_dir="${run_dirs[0]}"
      monitor_file="${LOG_ROOT}/${name}_gpu.csv"
      CUDA_VISIBLE_DEVICES="${ANT_GPU}" "${PYTHON}" \
        "${PROJECT_ROOT}/impls/summarize_coghp_ff_swiglu_204k.py" \
        --run-dir="${run_dir}" \
        --gpu-monitor="${monitor_file}" \
        --summary-csv="${summary_temp}" \
        >"${LOG_ROOT}/${name}_summary.log" 2>&1
    done
  done
  mv "${summary_temp}" "${SUMMARY_CSV}"
  echo "Summary CSV: ${SUMMARY_CSV}"
}

print_plan() {
  cat <<EOF
Model: CoGHP-FF-SwiGLU-RMSNorm-204k
Git commit: ${GIT_COMMIT_OVERRIDE:-<required>}
Code signature: ${CODE_SIGNATURE}
Noon HRM actual seeds: Ant {0,1,2}; Humanoid {0}
Requested seed policy: ${BASELINE_SEED_POLICY:-<required before preflight/launch>}
Output root: ${PROJECT_OUTPUT_ROOT}
Log root: ${LOG_ROOT}
Training: batch=256, updates=1M, eval=100k, eval episodes=50, save=1M
Action core: matched_feedforward, swiglu_rmsnorm, N=1, mi=256
Effective H/L cycles: N/A; gradient mode: N/A; calls B_L/B_H=1/1
Reference noon config: N1-H2L2-mi256 truncated, teacher chain.
Only model-forward change: action_core_type=hrm -> matched_feedforward.
EOF
}

case "${MODE}" in
  plan)
    print_plan
    ;;
  preflight)
    require_provenance
    set_seed_policy
    select_two_gpus
    mkdir -p "${PREFLIGHT_ROOT}" "${LOG_ROOT}"
    run_preflight_one antmaze_giant 0 "${ANT_GPU}"
    run_preflight_one humanoidmaze_giant 0 "${HUMANOID_GPU}"
    print_plan
    echo "Selected GPUs: Ant=${ANT_GPU}, Humanoid=${HUMANOID_GPU}"
    for seed in "${ANT_SEEDS[@]}"; do
      print_training_command antmaze_giant "${seed}" "${ANT_GPU}"
    done
    for seed in "${HUMANOID_SEEDS[@]}"; do
      print_training_command humanoidmaze_giant "${seed}" "${HUMANOID_GPU}"
    done
    ;;
  launch)
    require_provenance
    set_seed_policy
    select_two_gpus
    mkdir -p "${LOG_ROOT}"
    for seed in "${ANT_SEEDS[@]}"; do
      verify_preflight_one antmaze_giant "${seed}"
      assert_fresh_output "$(experiment_name antmaze_giant "${seed}")"
    done
    for seed in "${HUMANOID_SEEDS[@]}"; do
      verify_preflight_one humanoidmaze_giant "${seed}"
      assert_fresh_output "$(experiment_name humanoidmaze_giant "${seed}")"
    done
    if [[ -e "${SUMMARY_CSV}" ]]; then
      echo "Refusing to overwrite existing summary: ${SUMMARY_CSV}" >&2
      exit 2
    fi
    print_plan
    echo "Selected GPUs: Ant=${ANT_GPU}, Humanoid=${HUMANOID_GPU}"
    for seed in "${ANT_SEEDS[@]}"; do
      print_training_command antmaze_giant "${seed}" "${ANT_GPU}"
    done
    for seed in "${HUMANOID_SEEDS[@]}"; do
      print_training_command humanoidmaze_giant "${seed}" "${HUMANOID_GPU}"
    done
    run_task_queue antmaze_giant "${ANT_GPU}" "${ANT_SEEDS[@]}" &
    ant_queue_pid=$!
    run_task_queue humanoidmaze_giant "${HUMANOID_GPU}" \
      "${HUMANOID_SEEDS[@]}" &
    humanoid_queue_pid=$!
    ant_status=0
    humanoid_status=0
    wait "${ant_queue_pid}" || ant_status=$?
    wait "${humanoid_queue_pid}" || humanoid_status=$?
    if (( ant_status != 0 || humanoid_status != 0 )); then
      echo "Training queue failure: Ant=${ant_status}, Humanoid=${humanoid_status}." >&2
      exit 1
    fi
    summarize_all
    ;;
  summarize)
    require_provenance
    set_seed_policy
    select_two_gpus
    summarize_all
    ;;
  *)
    echo "Usage: $0 {plan|preflight|launch|summarize}" >&2
    exit 2
    ;;
esac
