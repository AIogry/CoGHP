#!/usr/bin/env bash
set -euo pipefail

# Train the shared-block Action-Core-CoGHP implementation.
#
# The action branch always performs exactly one outer core call.  The core
# topology and its shared H/L block family are selected independently.
# ACTION_CORE_TYPE supports vanilla_coghp, matched_feedforward,
# single_state_iterative, and hrm.  The vanilla path is the original
# CoGHP GELU MixerBlock stack and does not allocate HRM state.
#
# Example:
#   SEED=0 GPU_ID=0 ACTION_HRM_BACKBONE=mixer \
#     bash scripts/train_hrm_action_coghp.sh antmaze_giant

TASK="${1:-antmaze_giant}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-${ANT_GPU:-${HUMANOID_GPU:-0}}}"

ACTION_HRM_BACKBONE="${ACTION_HRM_BACKBONE:-mixer}"
ACTION_HRM_NUM_LAYERS="${ACTION_HRM_NUM_LAYERS:-1}"
ACTION_HRM_INTERMEDIATE_SIZE="${ACTION_HRM_INTERMEDIATE_SIZE:-512}"
ACTION_HRM_HEAD_DIM="${ACTION_HRM_HEAD_DIM:-32}"
ACTION_HRM_H_CYCLES="${ACTION_HRM_H_CYCLES:-2}"
ACTION_HRM_L_CYCLES="${ACTION_HRM_L_CYCLES:-2}"
ACTION_HRM_BPTT="${ACTION_HRM_BPTT:-True}"
ACTION_HRM_IS_CAUSAL="${ACTION_HRM_IS_CAUSAL:-True}"
ACTION_HRM_MIXER_INTERMEDIATE_SIZE="${ACTION_HRM_MIXER_INTERMEDIATE_SIZE:-256}"
ACTION_CORE_TYPE="${ACTION_CORE_TYPE:-hrm}"
ACTION_BLOCK_TYPE="${ACTION_BLOCK_TYPE:-}"
HRM_BLOCK_TYPE="${HRM_BLOCK_TYPE:-}"
ACTION_HRM_BLOCK_TYPE="${ACTION_HRM_BLOCK_TYPE:-}"
HRM_TOKEN_HIDDEN="${HRM_TOKEN_HIDDEN:-${ACTION_HRM_TOKEN_HIDDEN:-32}}"
HRM_CHANNEL_HIDDEN="${HRM_CHANNEL_HIDDEN:-${ACTION_HRM_CHANNEL_HIDDEN:-32}}"
HRM_MIXER_INTERMEDIATE_SIZE="${HRM_MIXER_INTERMEDIATE_SIZE:-${ACTION_HRM_MIXER_INTERMEDIATE_SIZE}}"
HRM_LAYERNORM_AFFINE="${HRM_LAYERNORM_AFFINE:-${ACTION_HRM_LAYERNORM_AFFINE:-True}}"
ACTION_HRM_NORM_EPS="${ACTION_HRM_NORM_EPS:-1e-6}"
ACTION_HRM_ROPE_THETA="${ACTION_HRM_ROPE_THETA:-10000.0}"
ACTION_HRM_FORWARD_DTYPE="${ACTION_HRM_FORWARD_DTYPE:-float32}"
ACTION_CHAIN_SOURCE="${ACTION_CHAIN_SOURCE:-teacher}"
RUN_TAG="${RUN_TAG:-}"
RUN_GROUP_OVERRIDE="${RUN_GROUP_OVERRIDE:-}"
CHECKPOINT_SIGNATURE_REQUIRED="${CHECKPOINT_SIGNATURE_REQUIRED:-False}"
EXPERIMENT_GIT_COMMIT="${EXPERIMENT_GIT_COMMIT:-}"
EXPERIMENT_CODE_SIGNATURE="${EXPERIMENT_CODE_SIGNATURE:-}"
EXPECTED_ACTION_CORE_PARAMS="${EXPECTED_ACTION_CORE_PARAMS:--1}"
PLANNER_CORE_TYPE="${PLANNER_CORE_TYPE:-vanilla_coghp}"
PLANNER_BLOCK_TYPE="${PLANNER_BLOCK_TYPE:-swiglu_rmsnorm}"
PLANNER_NUM_LAYERS="${PLANNER_NUM_LAYERS:-1}"
PLANNER_MIXER_INTERMEDIATE_SIZE="${PLANNER_MIXER_INTERMEDIATE_SIZE:-256}"
PLANNER_H_CYCLES="${PLANNER_H_CYCLES:-2}"
PLANNER_L_CYCLES="${PLANNER_L_CYCLES:-2}"
PLANNER_BPTT="${PLANNER_BPTT:-True}"
PLANNER_NORM_EPS="${PLANNER_NORM_EPS:-1e-6}"
PLANNER_FORWARD_DTYPE="${PLANNER_FORWARD_DTYPE:-float32}"
EXPECTED_PLANNER_CORE_PARAMS="${EXPECTED_PLANNER_CORE_PARAMS:--1}"
SHARE_ACTOR_CORE="${SHARE_ACTOR_CORE:-False}"


TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WANDB_MODE="${WANDB_MODE:-online}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/qijunrong/06-RL/offline-rl}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
OUTPUT_ROOT="${HRM_ACTION_OUTPUT_ROOT:-${DATA_ROOT}/exp}"
WANDB_PROJECT="${HRM_ACTION_WANDB_PROJECT:-HRMActionCoGHP}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/hrm_action_coghp}"
PYTHON="${PYTHON:-/home/eai/Tools/miniforge3/envs/brain_nav/bin/python}"
SKIP_DATASET_CHECK="${HRM_ACTION_SKIP_DATASET_CHECK:-0}"

if [[ -n "${HRM_BLOCK_TYPE}" && -n "${ACTION_HRM_BLOCK_TYPE}" \
  && "${HRM_BLOCK_TYPE}" != "${ACTION_HRM_BLOCK_TYPE}" ]]; then
  echo "HRM_BLOCK_TYPE conflicts with legacy ACTION_HRM_BLOCK_TYPE." >&2
  exit 2
fi
HRM_BLOCK_TYPE="${HRM_BLOCK_TYPE:-${ACTION_HRM_BLOCK_TYPE}}"

# Vanilla CoGHP has its own original MixerBlock implementation.  When the
# caller does not specify a block selector, choose that path before the
# generic Mixer/HRM selector fills in its SwiGLU default.
if [[ "${ACTION_CORE_TYPE}" == "vanilla_coghp" \
  && -z "${ACTION_BLOCK_TYPE}" \
  && ( -z "${HRM_BLOCK_TYPE}" || "${HRM_BLOCK_TYPE}" == "legacy" ) ]]; then
  ACTION_BLOCK_TYPE=coghp_gelu
fi

case "${PLANNER_CORE_TYPE}" in
  # legacy_coghp was the selector used by early planner experiments.  Keep
  # accepting it, but canonicalize it before writing the run configuration so
  # old commands retain exactly the vanilla CoGHP path.
  legacy_coghp) PLANNER_CORE_TYPE=vanilla_coghp ;;
  vanilla_coghp|matched_feedforward|hrm) ;;
  *)
    echo "PLANNER_CORE_TYPE must be vanilla_coghp, matched_feedforward, or hrm." >&2
    exit 2
    ;;
esac

case "${PLANNER_BLOCK_TYPE}" in
  swiglu_rmsnorm) ;;
  *)
    echo "PLANNER_BLOCK_TYPE currently supports only swiglu_rmsnorm." >&2
    exit 2
    ;;
esac

case "${PLANNER_FORWARD_DTYPE}" in
  float32|bfloat16) ;;
  *)
    echo "PLANNER_FORWARD_DTYPE must be float32 or bfloat16." >&2
    exit 2
    ;;
esac

if ! [[ "${PLANNER_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_NUM_LAYERS must be positive." >&2
  exit 2
fi

if ! [[ "${PLANNER_MIXER_INTERMEDIATE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_MIXER_INTERMEDIATE_SIZE must be positive." >&2
  exit 2
fi

for value in "${PLANNER_H_CYCLES}" "${PLANNER_L_CYCLES}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PLANNER_H_CYCLES and PLANNER_L_CYCLES must be positive." >&2
    exit 2
  fi
done

case "${PLANNER_BPTT}" in
  True|False) ;;
  *) echo "PLANNER_BPTT must be True or False." >&2; exit 2 ;;
esac
case "${SHARE_ACTOR_CORE}" in
  True|False) ;;
  *) echo "SHARE_ACTOR_CORE must be True or False." >&2; exit 2 ;;
esac
if [[ "${SHARE_ACTOR_CORE}" == "True" ]] && [[ "${PLANNER_CORE_TYPE}" != "hrm" || "${ACTION_CORE_TYPE}" != "hrm" ]]; then
  echo "SHARE_ACTOR_CORE=True requires PLANNER_CORE_TYPE=hrm and ACTION_CORE_TYPE=hrm." >&2
  exit 2
fi

if ! [[ "${EXPECTED_PLANNER_CORE_PARAMS}" =~ ^-?[0-9]+$ ]]; then
  echo "EXPECTED_PLANNER_CORE_PARAMS must be an integer." >&2
  exit 2
fi

case "${ACTION_CORE_TYPE}" in
  vanilla_coghp|matched_feedforward|single_state_iterative|hrm) ;;
  *)
    echo "ACTION_CORE_TYPE must be vanilla_coghp, matched_feedforward, single_state_iterative, or hrm." >&2
    exit 2
    ;;
esac
case "${ACTION_HRM_BACKBONE}" in
  mixer)
    ACTION_HRM_IS_MLP_MIXER=True
    if [[ -n "${ACTION_BLOCK_TYPE}" && -n "${HRM_BLOCK_TYPE}" \
      && "${HRM_BLOCK_TYPE}" != "legacy" \
      && "${ACTION_BLOCK_TYPE}" != "${HRM_BLOCK_TYPE}" ]]; then
      echo "ACTION_BLOCK_TYPE conflicts with legacy HRM_BLOCK_TYPE." >&2
      exit 2
    fi
    if [[ "${HRM_BLOCK_TYPE}" == "legacy" ]]; then
      HRM_BLOCK_TYPE=""
    fi
    ACTION_BLOCK_TYPE="${ACTION_BLOCK_TYPE:-${HRM_BLOCK_TYPE:-swiglu_rmsnorm}}"
    HRM_BLOCK_TYPE="${ACTION_BLOCK_TYPE}"
    ;;
  attention|attn)
    if [[ "${ACTION_CORE_TYPE}" != "hrm" ]]; then
      echo "Legacy attention is supported only with ACTION_CORE_TYPE=hrm." >&2
      exit 2
    fi
    ACTION_HRM_IS_MLP_MIXER=False
    ACTION_HRM_BACKBONE=attention
    if [[ -n "${ACTION_BLOCK_TYPE}" && "${ACTION_BLOCK_TYPE}" != "attention" ]]; then
      echo "ACTION_BLOCK_TYPE must be unset or attention for the legacy attention backbone." >&2
      exit 2
    fi
    if [[ -n "${HRM_BLOCK_TYPE}" && "${HRM_BLOCK_TYPE}" != "legacy" ]]; then
      echo "HRM_BLOCK_TYPE only selects Mixer blocks; leave it unset for attention." >&2
      exit 2
    fi
    ACTION_BLOCK_TYPE=attention
    HRM_BLOCK_TYPE=legacy
    ;;
  *) echo "ACTION_HRM_BACKBONE must be mixer or attention." >&2; exit 2 ;;
esac
if [[ "${ACTION_CORE_TYPE}" == "vanilla_coghp" ]]; then
  if [[ "${ACTION_HRM_BACKBONE}" != "mixer" ]]; then
    echo "Vanilla CoGHP executor requires ACTION_HRM_BACKBONE=mixer." >&2
    exit 2
  fi
  case "${ACTION_BLOCK_TYPE}" in
    auto|legacy|''|coghp_gelu) ;;
    *)
      echo "Vanilla CoGHP executor uses the original GELU MixerBlock; " \
        "do not select ${ACTION_BLOCK_TYPE}." >&2
      exit 2
      ;;
  esac
  # The actual vanilla path directly constructs utils.coghp_network.MixerBlock;
  # this selector is metadata only and is not sent to HRMCore.
  ACTION_BLOCK_TYPE=coghp_gelu
  HRM_BLOCK_TYPE=legacy
fi
if [[ "${ACTION_HRM_BACKBONE}" == "mixer" ]]; then
  case "${ACTION_BLOCK_TYPE}" in
    swiglu_rmsnorm|coghp_gelu|gelu_layernorm) ;;
    *)
      echo "ACTION_BLOCK_TYPE must be swiglu_rmsnorm, coghp_gelu, or gelu_layernorm." >&2
      exit 2
      ;;
  esac
fi
case "${ACTION_HRM_BPTT}" in
  True|False) ;;
  *) echo "ACTION_HRM_BPTT must be True or False." >&2; exit 2 ;;
esac
case "${ACTION_HRM_IS_CAUSAL}" in
  True|False) ;;
  *) echo "ACTION_HRM_IS_CAUSAL must be True or False." >&2; exit 2 ;;
esac
case "${HRM_LAYERNORM_AFFINE}" in
  True|False) ;;
  *) echo "HRM_LAYERNORM_AFFINE must be True or False." >&2; exit 2 ;;
esac
case "${ACTION_CHAIN_SOURCE}" in
  teacher|predicted) ;;
  *) echo "ACTION_CHAIN_SOURCE must be teacher or predicted." >&2; exit 2 ;;
esac
case "${CHECKPOINT_SIGNATURE_REQUIRED}" in
  True|False) ;;
  *) echo "CHECKPOINT_SIGNATURE_REQUIRED must be True or False." >&2; exit 2 ;;
esac
if [[ "${CHECKPOINT_SIGNATURE_REQUIRED}" == "True" \
  && -z "${EXPERIMENT_GIT_COMMIT}" ]]; then
  echo "EXPERIMENT_GIT_COMMIT is required for signed checkpoints." >&2
  exit 2
fi
if ! [[ "${EXPECTED_ACTION_CORE_PARAMS}" =~ ^-?[0-9]+$ ]]; then
  echo "EXPECTED_ACTION_CORE_PARAMS must be an integer." >&2
  exit 2
fi
case "${ACTION_HRM_FORWARD_DTYPE}" in
  float32|bfloat16) ;;
  *) echo "ACTION_HRM_FORWARD_DTYPE must be float32 or bfloat16." >&2; exit 2 ;;
esac
for value in \
  "${ACTION_HRM_NUM_LAYERS}" \
  "${ACTION_HRM_INTERMEDIATE_SIZE}" \
  "${ACTION_HRM_HEAD_DIM}" \
  "${ACTION_HRM_H_CYCLES}" \
  "${ACTION_HRM_L_CYCLES}" \
  "${ACTION_HRM_MIXER_INTERMEDIATE_SIZE}" \
  "${HRM_TOKEN_HIDDEN}" \
  "${HRM_CHANNEL_HIDDEN}" \
  "${HRM_MIXER_INTERMEDIATE_SIZE}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "HRM integer settings must all be positive; got ${value}." >&2
    exit 2
  fi
done

normalize_task() {
  case "$1" in
    pointmaze_medium|pointmaze-medium) printf '%s\n' pointmaze_medium ;;
    pointmaze_large|pointmaze-large) printf '%s\n' pointmaze_large ;;
    pointmaze_giant|pointmaze-giant) printf '%s\n' pointmaze_giant ;;
    antmaze_medium|antmaze-medium) printf '%s\n' antmaze_medium ;;
    antmaze_large|antmaze-large) printf '%s\n' antmaze_large ;;
    antmaze_giant|antmaze-giant) printf '%s\n' antmaze_giant ;;
    humanoidmaze_medium|humanoidmaze-medium) printf '%s\n' humanoidmaze_medium ;;
    humanoidmaze_large|humanoidmaze-large) printf '%s\n' humanoidmaze_large ;;
    humanoidmaze_giant|humanoidmaze-giant) printf '%s\n' humanoidmaze_giant ;;
    *) return 1 ;;
  esac
}

TASK_KEY="$(normalize_task "${TASK}")" || {
  echo "Unknown task ${TASK}; expected pointmaze/antmaze/humanoidmaze medium, large, or giant (use underscores)." >&2
  exit 2
}

case "${TASK_KEY}" in
  pointmaze_medium|pointmaze_large|pointmaze_giant|antmaze_medium|antmaze_large)
    ENV_NAME="${TASK_KEY//_/-}-navigate-v0"
    NUM_SUBGOALS=1
    SUBGOAL_STEPS=50
    FEATURE_DIM=128
    EXTRA_ARGS=()
    [[ "${TASK_KEY}" != *_medium ]] || SUBGOAL_STEPS=25
    [[ "${TASK_KEY}" != pointmaze_* ]] || FEATURE_DIM=32
    if [[ "${TASK_KEY}" == pointmaze_giant ]]; then
      NUM_SUBGOALS=3
      EXTRA_ARGS=(--agent.lr=1e-5)
    fi
    ;;
  antmaze_giant)
    ENV_NAME="antmaze-giant-navigate-v0"
    NUM_SUBGOALS=2
    SUBGOAL_STEPS=50
    FEATURE_DIM=128
    EXTRA_ARGS=()
    ;;
  humanoidmaze_medium|humanoidmaze_large|humanoidmaze_giant)
    ENV_NAME="${TASK_KEY//_/-}-navigate-v0"
    NUM_SUBGOALS=1
    [[ "${TASK_KEY}" != humanoidmaze_giant ]] || NUM_SUBGOALS=2
    SUBGOAL_STEPS=100
    FEATURE_DIM=128
    EXTRA_ARGS=(--agent.discount=0.995)
    ;;
esac

if [[ "${ACTION_HRM_BACKBONE}" == "attention" ]] \
  && (( FEATURE_DIM % ACTION_HRM_HEAD_DIM != 0 )); then
  echo "ACTION_HRM_HEAD_DIM must divide FEATURE_DIM=${FEATURE_DIM}." >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/impls:${PYTHONPATH:-}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ "${DETERMINISTIC:-1}" == "1" && "${XLA_FLAGS:-}" != *"--xla_gpu_deterministic_ops"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops"
fi

mkdir -p "${LOG_DIR}"
if [[ "${SKIP_DATASET_CHECK}" != "1" ]]; then
  (
    cd "${PROJECT_ROOT}/impls"
    "${PYTHON}" -c "
import os
import ogbench

project_root = os.path.abspath('${PROJECT_ROOT}')
dataset_dir = os.path.abspath('${DATASET_DIR}')
train_path = os.path.join(dataset_dir, '${ENV_NAME}.npz')
val_path = os.path.join(dataset_dir, '${ENV_NAME}-val.npz')
if not os.path.abspath(ogbench.__file__).startswith(project_root):
    raise SystemExit('ERROR: imported ogbench is not the local project copy')
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise SystemExit('ERROR: expected dataset files are missing')
print('Dataset check passed:', train_path, val_path)
"
  )
fi

if [[ "${ACTION_HRM_BPTT}" == "True" ]]; then
  bptt_tag="fullbptt"
else
  bptt_tag="truncated"
fi
if [[ "${ACTION_HRM_BACKBONE}" == "mixer" ]]; then
  case "${ACTION_BLOCK_TYPE}" in
    swiglu_rmsnorm)
      # Preserve the historical default run-group format.
      capacity_tag="mi${HRM_MIXER_INTERMEDIATE_SIZE}"
      ;;
    coghp_gelu)
      capacity_tag="cg_th${HRM_TOKEN_HIDDEN}_ch${HRM_CHANNEL_HIDDEN}"
      ;;
    gelu_layernorm)
      if [[ "${HRM_LAYERNORM_AFFINE}" == "True" ]]; then
        affine_tag="aff1"
      else
        affine_tag="aff0"
      fi
      capacity_tag="gln_th${HRM_TOKEN_HIDDEN}_ch${HRM_CHANNEL_HIDDEN}_${affine_tag}"
      ;;
  esac
else
  capacity_tag="ff${ACTION_HRM_INTERMEDIATE_SIZE}_hd${ACTION_HRM_HEAD_DIM}"
fi
case "${ACTION_CORE_TYPE}" in
  vanilla_coghp)
    RUN_GROUP="${TASK_KEY}_vanilla_coghp_action_mixer"
    RUN_GROUP+="_n${ACTION_HRM_NUM_LAYERS}_${ACTION_CHAIN_SOURCE}"
    low_block_calls=0
    high_block_calls=0
    ;;
  hrm)
    # Preserve all historical HRM run-group names.
    RUN_GROUP="${TASK_KEY}_hrm_action_${ACTION_HRM_BACKBONE}"
    RUN_GROUP+="_o1_n${ACTION_HRM_NUM_LAYERS}_h${ACTION_HRM_H_CYCLES}_l${ACTION_HRM_L_CYCLES}"
    RUN_GROUP+="_${capacity_tag}_${bptt_tag}_${ACTION_CHAIN_SOURCE}"
    low_block_calls=$((ACTION_HRM_H_CYCLES * ACTION_HRM_L_CYCLES))
    high_block_calls="${ACTION_HRM_H_CYCLES}"
    ;;
  single_state_iterative)
    RUN_GROUP="${TASK_KEY}_single_state_action_mixer"
    RUN_GROUP+="_o1_n${ACTION_HRM_NUM_LAYERS}_h${ACTION_HRM_H_CYCLES}_l${ACTION_HRM_L_CYCLES}"
    RUN_GROUP+="_${capacity_tag}_${bptt_tag}_${ACTION_CHAIN_SOURCE}"
    low_block_calls=$((ACTION_HRM_H_CYCLES * ACTION_HRM_L_CYCLES))
    high_block_calls="${ACTION_HRM_H_CYCLES}"
    ;;
  matched_feedforward)
    RUN_GROUP="${TASK_KEY}_matched_ff_action_mixer"
    RUN_GROUP+="_o1_n${ACTION_HRM_NUM_LAYERS}_${capacity_tag}"
    RUN_GROUP+="_${ACTION_CHAIN_SOURCE}"
    low_block_calls=1
    high_block_calls=1
    ;;
esac
if [[ "${SHARE_ACTOR_CORE}" == "True" ]]; then
  if [[ "${PLANNER_CORE_TYPE}" != "hrm" || "${ACTION_CORE_TYPE}" != "hrm" ]]; then
    echo "Internal error: shared actor core must use official HRM on both branches." >&2
    exit 2
  fi
  if [[ "${PLANNER_BPTT}" == "True" ]]; then
    shared_bptt_tag=fullbptt
  else
    shared_bptt_tag=truncated
  fi
  RUN_GROUP="${TASK_KEY}_shared_hrm_actor_h${PLANNER_H_CYCLES}_l${PLANNER_L_CYCLES}_${capacity_tag}_${shared_bptt_tag}_${ACTION_CHAIN_SOURCE}"
  low_block_calls=$((PLANNER_H_CYCLES * PLANNER_L_CYCLES))
  high_block_calls=${PLANNER_H_CYCLES}
fi
if [[ -n "${RUN_TAG}" ]]; then
  RUN_GROUP+="_${RUN_TAG}"
fi
if [[ -n "${RUN_GROUP_OVERRIDE}" ]]; then
  RUN_GROUP="${RUN_GROUP_OVERRIDE}"
fi
if [[ "${#RUN_GROUP}" -gt 128 ]]; then
  echo "Generated W&B group exceeds 128 characters: ${RUN_GROUP}" >&2
  exit 2
fi

echo "Starting ${RUN_GROUP} on GPU ${GPU_ID}"
echo "Action token sequence: [observation, goal, ${NUM_SUBGOALS} subgoals, action query]"
echo "Action core: ${ACTION_CORE_TYPE}; outer calls: 1; B_L/B_H calls: ${low_block_calls}/${high_block_calls}"
if [[ "${SHARE_ACTOR_CORE}" == "True" ]]; then
  echo "Planner/executor topology: one shared official HRM parameter tree; separate high/low readouts."
fi
echo "Action block: ${ACTION_BLOCK_TYPE} (token_hidden=${HRM_TOKEN_HIDDEN}, channel_hidden=${HRM_CHANNEL_HIDDEN}, mixer_intermediate=${HRM_MIXER_INTERMEDIATE_SIZE}, layernorm_affine=${HRM_LAYERNORM_AFFINE})"
if [[ "${ACTION_CORE_TYPE}" == "vanilla_coghp" ]]; then
  echo "Vanilla CoGHP executor: original GELU MixerBlock stack; num_mixer_blocks=${NUM_MIXER_BLOCKS:-1}"
fi
if [[ "${PLANNER_CORE_TYPE}" == "vanilla_coghp" ]]; then
  echo "Planner core: vanilla_coghp; shared CoGHP Mixer blocks=${NUM_MIXER_BLOCKS:-1}"
elif [[ "${PLANNER_CORE_TYPE}" == "matched_feedforward" ]]; then
  echo "Planner core: matched_feedforward; B_L/B_H calls per subgoal=1/1"
else
  echo "Planner core: official HRM; H=${PLANNER_H_CYCLES}, L=${PLANNER_L_CYCLES}, BPTT=${PLANNER_BPTT}; B_L/B_H calls per subgoal=$((PLANNER_H_CYCLES * PLANNER_L_CYCLES))/${PLANNER_H_CYCLES}"
fi
if [[ "${ACTION_CORE_TYPE}" == "matched_feedforward" ]]; then
  echo "Matched feed-forward ignores H/L cycle and BPTT settings by design."
fi
echo "Results: ${OUTPUT_ROOT}/${WANDB_PROJECT}/${RUN_GROUP}"
(
  cd "${PROJECT_ROOT}/impls"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" main.py \
    --run_group="${RUN_GROUP}" \
    --wandb_project="${WANDB_PROJECT}" \
    --wandb_mode="${WANDB_MODE}" \
    --env_name="${ENV_NAME}" \
    --save_dir="${OUTPUT_ROOT}" \
    --seed="${SEED}" \
    --agent=agents/hrm_action_coghp.py \
    --train_steps="${TRAIN_STEPS}" \
    --log_interval="${LOG_INTERVAL}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --eval_episodes="${EVAL_EPISODES}" \
    --eval_temperature=0 \
    --video_episodes=0 \
    --eval_on_cpu="${EVAL_ON_CPU}" \
    --checkpoint_signature_required="${CHECKPOINT_SIGNATURE_REQUIRED}" \
    --experiment_git_commit="${EXPERIMENT_GIT_COMMIT}" \
    --experiment_code_signature="${EXPERIMENT_CODE_SIGNATURE}" \
    --fixed_validation_batch=True \
    --require_deterministic_gpu=True \
    --agent.batch_size="${BATCH_SIZE}" \
    --agent.num_subgoals="${NUM_SUBGOALS}" \
    --agent.subgoal_steps="${SUBGOAL_STEPS}" \
    --agent.feature_dim="${FEATURE_DIM}" \
    --agent.planner_core_type="${PLANNER_CORE_TYPE}" \
    --agent.planner_block_type="${PLANNER_BLOCK_TYPE}" \
    --agent.planner_num_layers="${PLANNER_NUM_LAYERS}" \
    --agent.planner_mixer_intermediate_size="${PLANNER_MIXER_INTERMEDIATE_SIZE}" \
    --agent.planner_h_cycles="${PLANNER_H_CYCLES}" \
    --agent.planner_l_cycles="${PLANNER_L_CYCLES}" \
    --agent.planner_bptt="${PLANNER_BPTT}" \
    --agent.planner_norm_eps="${PLANNER_NORM_EPS}" \
    --agent.planner_forward_dtype="${PLANNER_FORWARD_DTYPE}" \
    --agent.share_actor_core="${SHARE_ACTOR_CORE}" \
    --agent.expected_planner_core_params="${EXPECTED_PLANNER_CORE_PARAMS}" \
    --agent.action_hrm_num_layers="${ACTION_HRM_NUM_LAYERS}" \
    --agent.action_hrm_intermediate_size="${ACTION_HRM_INTERMEDIATE_SIZE}" \
    --agent.action_hrm_head_dim="${ACTION_HRM_HEAD_DIM}" \
    --agent.action_hrm_h_cycles="${ACTION_HRM_H_CYCLES}" \
    --agent.action_hrm_l_cycles="${ACTION_HRM_L_CYCLES}" \
    --agent.action_hrm_bptt="${ACTION_HRM_BPTT}" \
    --agent.action_hrm_is_causal="${ACTION_HRM_IS_CAUSAL}" \
    --agent.action_hrm_is_mlp_mixer="${ACTION_HRM_IS_MLP_MIXER}" \
    --agent.action_hrm_mlp_mixer_intermediate_size="${ACTION_HRM_MIXER_INTERMEDIATE_SIZE}" \
    --agent.action_core_type="${ACTION_CORE_TYPE}" \
    --agent.action_block_type="${ACTION_BLOCK_TYPE}" \
    --agent.hrm_block_type="${HRM_BLOCK_TYPE}" \
    --agent.hrm_token_hidden="${HRM_TOKEN_HIDDEN}" \
    --agent.hrm_channel_hidden="${HRM_CHANNEL_HIDDEN}" \
    --agent.hrm_mixer_intermediate_size="${HRM_MIXER_INTERMEDIATE_SIZE}" \
    --agent.hrm_layernorm_affine="${HRM_LAYERNORM_AFFINE}" \
    --agent.expected_action_core_params="${EXPECTED_ACTION_CORE_PARAMS}" \
    --agent.action_hrm_norm_eps="${ACTION_HRM_NORM_EPS}" \
    --agent.action_hrm_rope_theta="${ACTION_HRM_ROPE_THETA}" \
    --agent.action_hrm_forward_dtype="${ACTION_HRM_FORWARD_DTYPE}" \
    --agent.train_action_chain_source="${ACTION_CHAIN_SOURCE}" \
    --agent.enable_action_hrm_validation=True \
    "${EXTRA_ARGS[@]}"
)
