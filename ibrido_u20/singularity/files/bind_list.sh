#!/bin/bash

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_20/singularity."
    exit
fi

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Check if PBS is installed
if command -v qstat >/dev/null 2>&1; then
    IS_PBS_AVAILABLE=true
else
    IS_PBS_AVAILABLE=false
fi

ME=$(whoami)

# defining base folder for framework
BASE_FOLDER="$HOME/work"
if [ "$IS_PBS_AVAILABLE" = true ]; then
    BASE_FOLDER="/fastwork/${ME}" # use fastwork for cluster
fi

# some definitions
SING_CONTAINER_DIR="$(dirname "$THIS_DIR")"
IBRIDO_PREFIX=$BASE_FOLDER/containers/ibrido-singularity-xbot
IBRIDO_WS_PREFIX=${IBRIDO_PREFIX}/ibrido_ws/
IBRIDO_WS_SRC=${IBRIDO_WS_PREFIX}/src
IBRIDO_CONDA=${IBRIDO_PREFIX}/conda

# defining files to be binded at runtime
IBRIDO_BFILES=(
    "${IBRIDO_PREFIX}/network/.netrc:/root/.netrc:rw"
)
# defining directories to be binded at runtime
IBRIDO_BDIRS=(
    "/dev/input:/dev/input:rw"
    "/tmp/.X11-unix:/tmp/.X11-unix"
    "/media/apatrizi/AP_ext/data/IBRIDO/:/root/IBRIDO"
    "${IBRIDO_CONTAINERS_PREFIX}/files:/root/ibrido_files"
    "${IBRIDO_CONTAINERS_PREFIX}/utils:/root/ibrido_utils"
    "${IBRIDO_PREFIX}/ibrido_logs:/root/ibrido_logs"
    "${IBRIDO_PREFIX}/tmp:/tmp:rw"
    "${IBRIDO_PREFIX}/config:/.config:rw"
    "${IBRIDO_PREFIX}/aux_data:/root/aux_data:rw"
    "${IBRIDO_PREFIX}/training_data:/root/training_data:rw"
    "${IBRIDO_WS_PREFIX}:/root/ibrido_ws:rw"
    "${IBRIDO_CONDA}:/opt/conda:rw"
    "${IBRIDO_PREFIX}/conda_hidden/.conda:/root/.conda:rw"
    "${IBRIDO_PREFIX}/.cache/wandb:/root/.cache/wandb:rw"
    "${IBRIDO_PREFIX}/.cache/pip:/root/.cache/pip:rw"
    "${IBRIDO_PREFIX}/.byobu:/root/.byobu:rw"
    "${IBRIDO_PREFIX}/.xbot:/root/.xbot:rw"
    "${IBRIDO_PREFIX}/.ros:/root/.ros:rw"
    "${IBRIDO_PREFIX}/.gazebo:/root/.gazebo:rw"
    "${IBRIDO_PREFIX}/.rviz2:/root/.rviz:rw"
)

OTHER_BDIRS=(
    ":"
)

# git directories and their branches
IBRIDO_GITDIRS=(
    "git@github.com:AndrePatri/IBRIDO.git*main"
    "git@github.com:AndrePatri/LRHControl.git*isaac4.0.0"
    "git@github.com:AndrePatri/CoClusterBridge.git*devel"
    "git@github.com:AndrePatri/EigenIPC.git*devel"
    "git@github.com:AndrePatri/LRHControlEnvs.git*isaac4.2.0"
    "git@github.com:AndrePatri/RHCViz.git*ros1_noetic"
    "git@github.com:AndrePatri/PerfSleep.git*main"
    "git@github.com:AndrePatri/phase_manager.git*add_update"
    "git@github.com:AndrePatri/unitree_ros.git*andrepatri_dev"
    "git@github.com:ADVRHumanoids/horizon.git*andrepatri_devel"
    "git@github.com:ADVRHumanoids/xbot2_mujoco.git*andrepatri_dev"
    "git@github.com:AndrePatri/mujoco_cmake.git*3.x"
    "git@github.com:ADVRHumanoids/KyonRLStepping.git*isaac4.0.0"
    "git@github.com:ADVRHumanoids/CentauroHybridMPC.git*isaac4.0.0"
    "git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git*big_wheels_v2.10_optional_find"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*optional_find"
    "git@github.com:AndrePatri/casadi.git*optional_float"
    "git@gitlab.com:crzz/adarl.git*andrepatri_dev"
    "git@gitlab.com:crzz/adarl_ros.git*andrepatri_dev"
    "git@gitlab.com:crzz/jumping_leg.git*andrepatri_dev"
)

# Concatenate
# IBRIDO_BDIRS=("${IBRIDO_BDIRS[@]}" "${OTHER_BDIRS[@]}")
IBRIDO_B_ALL=("${IBRIDO_BDIRS[@]}" "${IBRIDO_BFILES[@]}")

IBRIDO_BDIRS_SRC=()
# just extract binding SRC
for entry in "${IBRIDO_BDIRS[@]}"; do
    # Use parameter expansion to extract substring before the first colon
    filtered_entry="${entry%%:*}"
    IBRIDO_BDIRS_SRC+=("$filtered_entry")
done

IBRIDO_BFILES_SRC=()
# extracting bind files
for entry in "${IBRIDO_BFILES[@]}"; do
    # Use parameter expansion to extract substring before the first colon
    filtered_entry="${entry%%:*}"
    IBRIDO_BFILES_SRC+=("$filtered_entry")
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


