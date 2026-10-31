#!/usr/bin/env bash
# Complete CPU pipeline check on existing OGBench data; not a performance run.
# Usage: bash scripts/run_joint_planner_smoke.sh [RUN_DIR] [H_CYCLES] [L_CYCLES] [NUM_SUBGOALS]
# Default peripheral dimensions match v0. Set SMOKE_SMALL=1 for a cheaper
# (32,32)-MLP check; latent and both Mixer intermediate dimensions stay at 32.
set -euo pipefail

joint_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
joint_python="${PYTHON:-${HOME}/Tools/miniforge3/envs/brain_nav/bin/python}"
joint_h="${2:-2}"
joint_l="${3:-2}"
joint_subgoals="${4:-3}"
joint_run_dir="${1:-${joint_root}/impls/exp/joint_planner_v0_smoke_$(date +%Y%m%d_%H%M%S)_h${joint_h}l${joint_l}m${joint_subgoals}}"
joint_steps="${TRAIN_STEPS:-128}"
joint_small="${SMOKE_SMALL:-0}"
for joint_count in "${joint_h}" "${joint_l}" "${joint_subgoals}" "${joint_steps}"; do
    if [[ ! "${joint_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "H_CYCLES, L_CYCLES, NUM_SUBGOALS and TRAIN_STEPS must be positive integers." >&2
        exit 1
    fi
done
if [[ "${joint_small}" != 0 && "${joint_small}" != 1 ]]; then
    echo "SMOKE_SMALL must be 0 (full peripheral networks) or 1 (small smoke-only MLPs)." >&2
    exit 1
fi
joint_hidden='(512,512,512)'
if [[ "${joint_small}" == 1 ]]; then
    joint_hidden='(32,32)'
fi
joint_log_interval=32
if (( joint_steps < joint_log_interval )); then
    joint_log_interval="${joint_steps}"
fi
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/data/qijunrong/06-RL/offline-rl/data/raw_ogbench}"
export PYTHONPATH="${joint_root}:${joint_root}/impls:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"

# Avoid initiating a download when the smoke-test data path is misconfigured.
for joint_suffix in '' '-val'; do
    if [[ ! -f "${OGBENCH_DATASET_DIR}/pointmaze-medium-navigate-v0${joint_suffix}.npz" ]]; then
        echo "Missing dataset: ${OGBENCH_DATASET_DIR}/pointmaze-medium-navigate-v0${joint_suffix}.npz" >&2
        exit 1
    fi
done

printf 'Joint planner v0 CPU check: h=%s l=%s M=%s, external K=1, Mixer widths=32, latent=32, peripheral MLPs=%s, batch=16, updates=%s\n' \
    "${joint_h}" "${joint_l}" "${joint_subgoals}" "${joint_hidden}" "${joint_steps}"
cd "${joint_root}"
exec "${joint_python}" impls/main.py \
    --agent="${joint_root}/impls/agents/navigation.py:joint_planner" \
    --env_name=pointmaze-medium-navigate-v0 \
    --run_dir="${joint_run_dir}" \
    --wandb_mode=disabled \
    --seed=0 \
    --train_steps="${joint_steps}" \
    --log_interval="${joint_log_interval}" \
    --eval_interval="${joint_steps}" \
    --save_interval="${joint_steps}" \
    --eval_tasks=1 --eval_episodes=1 --video_episodes=0 \
    --agent.batch_size=16 \
    --agent.actor_hidden_dims="${joint_hidden}" \
    --agent.value_hidden_dims="${joint_hidden}" \
    --agent.enc_hidden_dims="${joint_hidden}" \
    --agent.feature_dim=32 \
    --agent.mixer_hidden=32 \
    --agent.planner_mixer_intermediate_size=32 \
    --agent.planner_refinement_steps=1 \
    --agent.planner_h_cycles="${joint_h}" \
    --agent.planner_l_cycles="${joint_l}" \
    --agent.num_subgoals="${joint_subgoals}"
