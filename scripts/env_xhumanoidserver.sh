#!/usr/bin/env bash
# Source this file on xhumanoidserver before launching experiments:
#   source scripts/env_xhumanoidserver.sh

export DATA_ROOT="${DATA_ROOT:-/nix/data/offline-rl}"
export DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/data/raw_ogbench}"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-${DATASET_DIR}}"
export PYTHON="${PYTHON:-/root/Tools/miniforge3/envs/brain_nav/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
