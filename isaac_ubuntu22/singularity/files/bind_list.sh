#!/bin/bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# some definitions
SING_CONTAINER_DIR="$(dirname "$THIS_DIR")"
IBRIDO_PREFIX=$HOME/containers/ibrido-singularity
IBRIDO_WS_SRC=${IBRIDO_PREFIX}/ibrido_ws/src

# defining directories to be binded at runtime
IBRIDO_BDIRS=(
    "${IBRIDO_PREFIX}/aux_data:/root/aux_data:rw"
    "${IBRIDO_PREFIX}/training_data:/root/training_data:rw"
    "${IBRIDO_PREFIX}/ibrido_ws:/root/ibrido_ws:rw"
    "${IBRIDO_PREFIX}/conda:/opt/conda:rw"
    "${IBRIDO_PREFIX}/conda_hidden/.conda:/root/.conda:rw"
    "${IBRIDO_PREFIX}/.cache/wandb:/root/.cache/wandb:rw"
    "${IBRIDO_PREFIX}/network/.netrc:/root/.netrc:rw"
    "${IBRIDO_PREFIX}/.byobu:/root/.byobu:rw"
)

ISAAC_BDIRS=(
    "${IBRIDO_PREFIX}/isaac-sim/cache/ov:/root/.cache/ov:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/pip:/root/.cache/pip:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/warp:/root/.cache/warp:rw"
    "${IBRIDO_PREFIX}/isaac-sim/local:/root/.local:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw"
    "${IBRIDO_PREFIX}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw"
    "${IBRIDO_PREFIX}/isaac-sim/documents:/root/Documents:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/nv_shadercache:/isaac-sim/kit/exts/omni.gpu_foundation/cache/nv_shadercache:rw"
    "${IBRIDO_PREFIX}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw"
    "${IBRIDO_PREFIX}/isaac-sim/data:/isaac-sim/kit/data:rw"
    "${IBRIDO_PREFIX}/isaac-sim/kitlogs/Isaac-Sim:/isaac-sim/kit/logs/Kit/Isaac-Sim:rw"
)

# git directories and their branches
IBRIDO_GITDIRS=(
    "git@github.com:AndrePatri/IBRIDO.git*main"
    "git@github.com:AndrePatri/LRHControl.git*isaac4.0.0"
    "git@github.com:AndrePatri/CoClusterBridge.git*devel"
    "git@github.com:AndrePatri/SharsorIPCpp.git*devel"
    "git@github.com:AndrePatri/OmniRoboGym.git*isaac4.0.0"
    "git@github.com:AndrePatri/RHCViz.git*ros2_humble"
    "git@github.com:AndrePatri/phase_manager.git*new_architecture"
    "git@github.com:AndrePatri/unitree_ros.git*andrepatri_dev"
    "git@github.com:ADVRHumanoids/horizon.git*andrepatri_devel"
    "git@github.com:ADVRHumanoids/KyonRLStepping.git*isaac4.0.0"
    "git@github.com:ADVRHumanoids/CentauroHybridMPC.git*isaac4.0.0"
    "git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git*big_wheels_v2.10_optional_find_ros2"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*optional_find_ros2"
)

# Concatenate the arrays
IBRIDO_BDIRS=("${IBRIDO_BDIRS[@]}" "${ISAAC_BDIRS[@]}")

IBRIDO_BDIRS_SRC=()
# just extract binding SRC
for entry in "${IBRIDO_BDIRS[@]}"; do
    # Use parameter expansion to extract substring before the first colon
    filtered_entry="${entry%%:*}"
    IBRIDO_BDIRS_SRC+=("$filtered_entry")
done

# extract git repo info
IBRIDO_GIT_SRC=()
IBRIDO_GIT_BRCH=()
for entry in "${IBRIDO_GITDIRS[@]}"; do
    # Split entry based on colon
    IFS='*' read -r src branch <<< "$entry"
    
    # Add to respective arrays
    IBRIDO_GIT_SRC+=("$src")
    IBRIDO_GIT_BRCH+=("$branch")
done
