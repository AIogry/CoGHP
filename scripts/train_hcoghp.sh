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
# Segment reset versus detached carry on AntMaze-Giant, one mode per GPU:
#   SEED=0 RESET_GPU=0 CARRY_GPU=1 \
#     bash scripts/train_hcoghp.sh state_init_compare
#
# All three Giant tasks for one seed on a two-GPU machine:
#   SEED=0 POINT_GPU=0 ANT_GPU=1 HUMANOID_GPU=0 \
#     bash scripts/train_hcoghp.sh giant
#
# New HRM initialization can be overridden explicitly:
#   HCOGHP_L_CYCLES=2 HCOGHP_INIT_STD=0.1 \
#   HCOGHP_TRAINABLE_INIT=True HCOGHP_RESIDUAL_ALPHA=0.05 \
#     bash scripts/train_hcoghp.sh antmaze_large
#
# Eight-way A800 architecture screen on AntMaze-Giant (one run per GPU):
#   SEED=0 A800_GPUS="0 1 2 3 4 5 6 7" \
#     bash scripts/train_hcoghp.sh a800_residual_sweep
#
# Local two-GPU L-cycle screen on AntMaze-Giant.  GPU A runs L=2 then L=3;
# GPU B runs L=4, which approximately balances the total recurrent compute:
#   SEED=0 L_SWEEP_GPU_A=0 L_SWEEP_GPU_B=1 \
#     bash scripts/train_hcoghp.sh local_l_sweep

TASK="${1:-giant}"
SEED="${SEED:-0}"
RUN_ID="sd$(printf '%03d' "${SEED}")_$(date +%Y%m%d_%H%M%S)"

H_CYCLES="${HCOGHP_H_CYCLES:-2}"
L_CYCLES="${HCOGHP_L_CYCLES:-2}"
INIT_STD="${HCOGHP_INIT_STD:-0.1}"
TRAINABLE_INIT="${HCOGHP_TRAINABLE_INIT:-True}"
LOW_NUM_BLOCKS="${HCOGHP_LOW_NUM_BLOCKS:-1}"
HIGH_NUM_BLOCKS="${HCOGHP_HIGH_NUM_BLOCKS:-1}"
READOUT_HIDDEN_DIM="${HCOGHP_READOUT_HIDDEN_DIM:-0}"
READOUT_MODE="${HCOGHP_READOUT_MODE:-residual}"
RESIDUAL_ALPHA="${HCOGHP_RESIDUAL_ALPHA:-0.05}"
STAGE_EMBED_CONTEXT="${HCOGHP_STAGE_EMBED_CONTEXT:-False}"
STATE_INIT_MODE="${HCOGHP_SEGMENT_STATE_INIT_MODE:-carry}"
ENABLE_DIAGNOSTICS="${HCOGHP_DIAGNOSTICS:-0}"
ENABLE_FREE_VALIDATION="${HCOGHP_FREE_VALIDATION:-0}"
ENABLE_NOISY_FREE_VALIDATION="${HCOGHP_NOISY_FREE_VALIDATION:-0}"
NOISY_FREE_TEMPERATURE="${HCOGHP_NOISY_FREE_TEMPERATURE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

POINT_GPU="${POINT_GPU:-0}"
ANT_GPU="${ANT_GPU:-1}"
HUMANOID_GPU="${HUMANOID_GPU:-0}"
RESET_GPU="${RESET_GPU:-0}"
CARRY_GPU="${CARRY_GPU:-1}"
A800_GPUS="${A800_GPUS:-0 1 2 3 4 5 6 7}"
L_SWEEP_GPU_A="${L_SWEEP_GPU_A:-0}"
L_SWEEP_GPU_B="${L_SWEEP_GPU_B:-1}"
SKIP_DATASET_CHECK="${HCOGHP_SKIP_DATASET_CHECK:-0}"
BATCH_PIDS=()

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
IMPLS_DIR="${PROJECT_ROOT}/impls"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/hcoghp}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
OUTPUT_ROOT="${HCOGHP_OUTPUT_ROOT:-${DATA_ROOT}/exp}"
WANDB_PROJECT="${HCOGHP_WANDB_PROJECT:-HCoGHP}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}:${IMPLS_DIR}:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ "${DETERMINISTIC:-1}" == "1" ]]; then
  if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_deterministic_ops"* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops"
  fi
fi

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --train_steps="${TRAIN_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --eval_episodes="${EVAL_EPISODES}"
  --eval_temperature=0
  --video_episodes=0
  --eval_on_cpu="${EVAL_ON_CPU}"
  --fixed_validation_batch=True
  --require_deterministic_gpu=True
  --seed="${SEED}"
  --agent=agents/hcoghp.py
  --save_dir="${OUTPUT_ROOT}"
  --wandb_project="${WANDB_PROJECT}"
  --wandb_mode="${WANDB_MODE}"
  --agent.hrm_h_cycles="${H_CYCLES}"
  --agent.hrm_l_cycles="${L_CYCLES}"
  --agent.hrm_init_std="${INIT_STD}"
  --agent.hrm_trainable_init="${TRAINABLE_INIT}"
  --agent.hrm_low_num_blocks="${LOW_NUM_BLOCKS}"
  --agent.hrm_high_num_blocks="${HIGH_NUM_BLOCKS}"
  --agent.subgoal_readout_hidden_dim="${READOUT_HIDDEN_DIM}"
  --agent.hrm_readout_mode="${READOUT_MODE}"
  --agent.hrm_readout_residual_alpha="${RESIDUAL_ALPHA}"
  --agent.stage_embedding_in_context="${STAGE_EMBED_CONTEXT}"
  --agent.batch_size="${BATCH_SIZE}"
  # HCoGHP training always receives dataset subgoal targets (teacher forcing).
  --agent.scheduled_sampling_enabled=False
  --agent.scheduled_sampling_max_prob=0.0
  --agent.scheduled_sampling_stop_gradient=True
)

if [[ "${ENABLE_DIAGNOSTICS}" == "1" ]]; then
  COMMON_ARGS+=(--agent.enable_hcoghp_diagnostics=True)
fi

if [[ "${ENABLE_FREE_VALIDATION}" == "1" ]]; then
  COMMON_ARGS+=(--agent.enable_free_running_validation=True)
fi

if [[ "${ENABLE_NOISY_FREE_VALIDATION}" == "1" ]]; then
  COMMON_ARGS+=(
    --agent.enable_noisy_free_running_validation=True
    --agent.noisy_free_running_temperature="${NOISY_FREE_TEMPERATURE}"
  )
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
  local state_init_mode="${3:-${STATE_INIT_MODE}}"
  local spec env_name num_subgoals subgoal_steps feature_dim extra_args run_group
  local init_tag trainable_tag readout_tag readout_mode_tag residual_tag state_tag
  local stage_context_tag diagnostics_tag free_validation_tag

  spec="$(task_spec_for "${task_key}")" || {
    echo "Unknown task: ${task_key}" >&2
    return 2
  }

  IFS='|' read -r task_key env_name num_subgoals subgoal_steps feature_dim extra_args <<< "${spec}"
  if [[ "${state_init_mode}" != "reset" && "${state_init_mode}" != "carry" ]]; then
    echo "Unknown HCoGHP segment state initialization mode: ${state_init_mode}" >&2
    return 2
  fi

  if [[ "${SKIP_DATASET_CHECK}" != "1" ]]; then
    check_dataset_runtime "${env_name}"
  fi
  init_tag="${INIT_STD//./p}"
  case "${TRAINABLE_INIT,,}" in
    true|1) trainable_tag="1" ;;
    false|0) trainable_tag="0" ;;
    *)
      echo "HCOGHP_TRAINABLE_INIT must be True/False or 1/0." >&2
      return 2
      ;;
  esac
  if [[ "${READOUT_HIDDEN_DIM}" == "0" ]]; then
    readout_tag="2x"
  else
    readout_tag="${READOUT_HIDDEN_DIM}"
  fi
  residual_tag="${RESIDUAL_ALPHA//./p}"
  case "${STAGE_EMBED_CONTEXT,,}" in
    true|1) stage_context_tag="1" ;;
    false|0) stage_context_tag="0" ;;
    *)
      echo "HCOGHP_STAGE_EMBED_CONTEXT must be True/False or 1/0." >&2
      return 2
      ;;
  esac
  if [[ "${READOUT_MODE}" != "residual" && "${READOUT_MODE}" != "replace" ]]; then
    echo "HCOGHP_READOUT_MODE must be residual or replace." >&2
    return 2
  fi
  if [[ "${READOUT_MODE}" == "residual" ]]; then
    readout_mode_tag="res"
  else
    readout_mode_tag="rep"
  fi
  if [[ "${state_init_mode}" == "carry" ]]; then
    state_tag="car"
  else
    state_tag="rst"
  fi
  diagnostics_tag="$([[ "${ENABLE_DIAGNOSTICS}" == "1" ]] && printf 1 || printf 0)"
  free_validation_tag="$([[ "${ENABLE_FREE_VALIDATION}" == "1" ]] && printf 1 || printf 0)"

  # W&B limits GroupName to 128 characters.  Keep every experiment-defining
  # field, but use compact, documented tags so the W&B group and result path
  # remain readable and unique without truncation.
  run_group="${task_key}_hcoghp_${state_tag}"
  run_group+="_k${num_subgoals}_s${subgoal_steps}_f${feature_dim}"
  run_group+="_h${H_CYCLES}_l${L_CYCLES}_b${LOW_NUM_BLOCKS}x${HIGH_NUM_BLOCKS}"
  run_group+="_r${readout_tag}_${readout_mode_tag}_a${residual_tag}_se${stage_context_tag}"
  run_group+="_bs${BATCH_SIZE}_i${init_tag}_ti${trainable_tag}"
  run_group+="_tf1_ss0_d${diagnostics_tag}_fv${free_validation_tag}${RUN_SUFFIX}"
  if [[ "${#run_group}" -gt 128 ]]; then
    echo "Generated W&B group name is ${#run_group} characters (limit: 128):" >&2
    echo "${run_group}" >&2
    echo "Shorten RUN_SUFFIX before launching this experiment." >&2
    return 2
  fi

  (
    cd "${IMPLS_DIR}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON}" main.py \
      --run_group="${run_group}" \
      --env_name="${env_name}" \
      "${COMMON_ARGS[@]}" \
      --agent.segment_state_init_mode="${state_init_mode}" \
      --agent.segment_carry_gradient_mode=detached \
      --agent.num_subgoals="${num_subgoals}" \
      --agent.subgoal_steps="${subgoal_steps}" \
      --agent.feature_dim="${feature_dim}" \
      ${extra_args}
  )
}

run_state_init_compare() {
  local reset_log carry_log reset_status carry_status

  if [[ "${RESET_GPU}" == "${CARRY_GPU}" ]]; then
    echo "RESET_GPU and CARRY_GPU must identify different GPUs." >&2
    return 2
  fi

  reset_log="${LOG_DIR}/antmaze_giant_segment_reset_${RUN_ID}.log"
  carry_log="${LOG_DIR}/antmaze_giant_segment_carry_${RUN_ID}.log"

  echo "Starting matched HCoGHP segment-state comparison for seed ${SEED}."
  echo "Segment reset:          GPU ${RESET_GPU}"
  echo "Segment detached carry: GPU ${CARRY_GPU}"
  echo "Results root: ${OUTPUT_ROOT}/${WANDB_PROJECT}"
  echo "Shared parameters: K=2, subgoal_steps=50, feature_dim=128,"
  echo "  H-cycles=${H_CYCLES}, L-cycles=${L_CYCLES}, low_blocks=${LOW_NUM_BLOCKS}, high_blocks=${HIGH_NUM_BLOCKS},"
  echo "  readout=${READOUT_HIDDEN_DIM} (0=2x feature dim), batch_size=${BATCH_SIZE},"
  echo "  readout_mode=${READOUT_MODE}, residual_alpha=${RESIDUAL_ALPHA},"
  echo "  stage_embedding_in_context=${STAGE_EMBED_CONTEXT},"
  echo "  teacher_forcing=True, scheduled_sampling=False, init_std=${INIT_STD},"
  echo "  trainable_init=${TRAINABLE_INIT}, train_steps=${TRAIN_STEPS}."
  echo "Logs:"
  echo "  ${reset_log}"
  echo "  ${carry_log}"

  run_task antmaze_giant "${RESET_GPU}" reset >"${reset_log}" 2>&1 &
  RESET_PID=$!
  run_task antmaze_giant "${CARRY_GPU}" carry >"${carry_log}" 2>&1 &
  CARRY_PID=$!

  echo "Segment reset PID: ${RESET_PID}"
  echo "Segment carry PID: ${CARRY_PID}"

  set +e
  wait "${RESET_PID}"
  reset_status=$?
  wait "${CARRY_PID}"
  carry_status=$?
  set -e
  RESET_PID=""
  CARRY_PID=""

  if [[ "${reset_status}" -ne 0 || "${carry_status}" -ne 0 ]]; then
    echo "At least one HCoGHP comparison run failed." >&2
    echo "Segment reset exit code: ${reset_status}" >&2
    echo "Segment carry exit code: ${carry_status}" >&2
    echo "Inspect ${reset_log} and ${carry_log}." >&2
    return 1
  fi

  echo "Both HCoGHP comparison runs completed successfully."
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

run_antmaze_variant() {
  local gpu_id="$1"
  local l_cycles="$2"
  local readout_mode="$3"
  local residual_alpha="$4"
  local trainable_init="$5"
  local stage_embed_context="$6"
  local state_init_mode="$7"
  local run_suffix="$8"

  env \
    SEED="${SEED}" \
    ANT_GPU="${gpu_id}" \
    DATA_ROOT="${DATA_ROOT}" \
    DATASET_DIR="${DATASET_DIR}" \
    HCOGHP_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HCOGHP_WANDB_PROJECT="${WANDB_PROJECT}" \
    LOG_DIR="${LOG_DIR}" \
    PYTHON="${PYTHON}" \
    WANDB_MODE="${WANDB_MODE}" \
    HCOGHP_H_CYCLES=2 \
    HCOGHP_L_CYCLES="${l_cycles}" \
    HCOGHP_INIT_STD="${INIT_STD}" \
    HCOGHP_TRAINABLE_INIT="${trainable_init}" \
    HCOGHP_LOW_NUM_BLOCKS="${LOW_NUM_BLOCKS}" \
    HCOGHP_HIGH_NUM_BLOCKS="${HIGH_NUM_BLOCKS}" \
    HCOGHP_READOUT_HIDDEN_DIM="${READOUT_HIDDEN_DIM}" \
    HCOGHP_READOUT_MODE="${readout_mode}" \
    HCOGHP_RESIDUAL_ALPHA="${residual_alpha}" \
    HCOGHP_STAGE_EMBED_CONTEXT="${stage_embed_context}" \
    HCOGHP_SEGMENT_STATE_INIT_MODE="${state_init_mode}" \
    HCOGHP_DIAGNOSTICS=1 \
    HCOGHP_FREE_VALIDATION=1 \
    HCOGHP_NOISY_FREE_VALIDATION=0 \
    HCOGHP_SKIP_DATASET_CHECK=1 \
    BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    EVAL_ON_CPU="${EVAL_ON_CPU}" \
    RUN_SUFFIX="${run_suffix}" \
    bash "${BASH_SOURCE[0]}" antmaze_giant
}

wait_for_batch() {
  local batch_name="$1"
  shift
  local -a pids=("$@")
  local status=0
  local pid exit_code

  set +e
  for pid in "${pids[@]}"; do
    wait "${pid}"
    exit_code=$?
    if [[ "${exit_code}" -ne 0 ]]; then
      echo "${batch_name}: worker PID ${pid} exited with code ${exit_code}." >&2
      status=1
    fi
  done
  set -e
  BATCH_PIDS=()

  if [[ "${status}" -ne 0 ]]; then
    echo "${batch_name} did not complete successfully; inspect the logs listed above." >&2
    return 1
  fi
  echo "${batch_name} completed successfully."
}

run_a800_residual_sweep() {
  local -a gpus labels readout_modes alphas trainable_inits stage_contexts state_modes
  local -a pids logs
  local i log

  read -r -a gpus <<< "${A800_GPUS}"
  if [[ "${#gpus[@]}" -ne 8 ]]; then
    echo "A800_GPUS must contain exactly eight space-separated GPU IDs." >&2
    return 2
  fi

  labels=(
    alpha000 alpha001 alpha002 default_alpha005
    fixed_init stage_in_context replace_readout reset_state
  )
  readout_modes=(residual residual residual residual residual residual replace residual)
  alphas=(0.0 0.01 0.02 0.05 0.05 0.05 0.05 0.05)
  trainable_inits=(True True True True False True True True)
  stage_contexts=(False False False False False True False False)
  state_modes=(carry carry carry carry carry carry carry reset)

  check_dataset_runtime "antmaze-giant-navigate-v0"
  echo "Starting the eight-way HCoGHP A800 screen on AntMaze-Giant, seed ${SEED}."
  echo "Shared setting: K=2, subgoal_steps=50, feature_dim=128, H=2, L=2,"
  echo "  batch_size=${BATCH_SIZE}, train_steps=${TRAIN_STEPS}, teacher forcing, no scheduled sampling,"
  echo "  deterministic evaluation temperature=0, diagnostics and free-running validation enabled."
  echo "Results root: ${OUTPUT_ROOT}/${WANDB_PROJECT}"

  for i in "${!gpus[@]}"; do
    log="${LOG_DIR}/antmaze_giant_a800_${labels[$i]}_${RUN_ID}.log"
    logs[$i]="${log}"
    run_antmaze_variant \
      "${gpus[$i]}" 2 "${readout_modes[$i]}" "${alphas[$i]}" \
      "${trainable_inits[$i]}" "${stage_contexts[$i]}" "${state_modes[$i]}" \
      "_a800_${labels[$i]}" >"${log}" 2>&1 &
    pids[$i]=$!
  done
  BATCH_PIDS=("${pids[@]}")

  echo "Launched configurations:"
  for i in "${!gpus[@]}"; do
    echo "  GPU ${gpus[$i]} PID ${pids[$i]}: ${labels[$i]} | readout=${readout_modes[$i]}, alpha=${alphas[$i]}, trainable_init=${trainable_inits[$i]}, stage_in_context=${stage_contexts[$i]}, state=${state_modes[$i]}"
    echo "    log: ${logs[$i]}"
  done

  wait_for_batch "A800 HCoGHP architecture screen" "${pids[@]}"
}

run_local_l_sweep() {
  local l2_log l3_log l4_log worker_a_status worker_b_status
  local worker_a_pid worker_b_pid

  if [[ "${L_SWEEP_GPU_A}" == "${L_SWEEP_GPU_B}" ]]; then
    echo "L_SWEEP_GPU_A and L_SWEEP_GPU_B must identify different GPUs." >&2
    return 2
  fi

  check_dataset_runtime "antmaze-giant-navigate-v0"
  l2_log="${LOG_DIR}/antmaze_giant_local_l2_control_${RUN_ID}.log"
  l3_log="${LOG_DIR}/antmaze_giant_local_l3_${RUN_ID}.log"
  l4_log="${LOG_DIR}/antmaze_giant_local_l4_${RUN_ID}.log"

  echo "Starting the local HCoGHP L-cycle screen on AntMaze-Giant, seed ${SEED}."
  echo "GPU ${L_SWEEP_GPU_A}: L=2 local control, then L=3."
  echo "GPU ${L_SWEEP_GPU_B}: L=4."
  echo "Shared setting: K=2, subgoal_steps=50, feature_dim=128, H=2, residual readout alpha=0.05,"
  echo "  trainable init, stage embedding outside the context mixer, detached carry,"
  echo "  batch_size=${BATCH_SIZE}, train_steps=${TRAIN_STEPS}, teacher forcing, no scheduled sampling,"
  echo "  deterministic evaluation temperature=0, diagnostics and free-running validation enabled."
  echo "Results root: ${OUTPUT_ROOT}/${WANDB_PROJECT}"
  echo "Logs:"
  echo "  ${l2_log}"
  echo "  ${l3_log}"
  echo "  ${l4_log}"

  (
    run_antmaze_variant "${L_SWEEP_GPU_A}" 2 residual 0.05 True False carry \
      _local_l2_control >"${l2_log}" 2>&1
    run_antmaze_variant "${L_SWEEP_GPU_A}" 3 residual 0.05 True False carry \
      _local_l3 >"${l3_log}" 2>&1
  ) &
  worker_a_pid=$!

  run_antmaze_variant "${L_SWEEP_GPU_B}" 4 residual 0.05 True False carry \
    _local_l4 >"${l4_log}" 2>&1 &
  worker_b_pid=$!
  BATCH_PIDS=("${worker_a_pid}" "${worker_b_pid}")

  echo "Worker L=2→L=3 PID: ${worker_a_pid}"
  echo "Worker L=4 PID: ${worker_b_pid}"

  set +e
  wait "${worker_a_pid}"
  worker_a_status=$?
  wait "${worker_b_pid}"
  worker_b_status=$?
  set -e
  BATCH_PIDS=()

  if [[ "${worker_a_status}" -ne 0 || "${worker_b_status}" -ne 0 ]]; then
    echo "At least one local L-cycle worker failed." >&2
    echo "L=2→L=3 worker exit code: ${worker_a_status}" >&2
    echo "L=4 worker exit code: ${worker_b_status}" >&2
    return 1
  fi
  echo "Local HCoGHP L-cycle screen completed successfully."
}

cleanup() {
  if [[ -n "${RESET_PID:-}" ]]; then
    kill "${RESET_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CARRY_PID:-}" ]]; then
    kill "${CARRY_PID}" 2>/dev/null || true
  fi
  if [[ "${#BATCH_PIDS[@]}" -gt 0 ]]; then
    local pid
    for pid in "${BATCH_PIDS[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
  fi
}
trap cleanup INT TERM

case "${TASK}" in
  compare|state_init_compare|antmaze_giant_state_init_compare)
    run_state_init_compare
    ;;
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
  a800_residual_sweep|residual_sweep_a800)
    run_a800_residual_sweep
    ;;
  local_l_sweep|l_sweep_local)
    run_local_l_sweep
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Expected one of: compare, state_init_compare, antmaze_giant_state_init_compare," >&2
    echo "  large, antmaze_large, pointmaze_giant, antmaze_giant," >&2
    echo "  humanoidmaze_giant, giant, all, a800_residual_sweep, local_l_sweep" >&2
    exit 2
    ;;
esac
