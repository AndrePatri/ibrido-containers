#!/bin/bash

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
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
IBRIDO_PREFIX=$BASE_FOLDER/containers/ibrido-singularity
IBRIDO_WS_PREFIX=${IBRIDO_PREFIX}/ibrido_ws/
IBRIDO_WS_SRC=${IBRIDO_WS_PREFIX}/src
IBRIDO_CONDA=${IBRIDO_PREFIX}/conda

# defining files to be binded at runtime
IBRIDO_BFILES=(
    "${IBRIDO_PREFIX}/network/.netrc:/root/.netrc:rw"
    "/etc/localtime:/etc/localtime:ro"
)
# defining directories to be binded at runtime
IBRIDO_BDIRS=(
    "${IBRIDO_CONTAINERS_PREFIX}/files:/root/ibrido_files"
    "${IBRIDO_CONTAINERS_PREFIX}/utils:/root/ibrido_utils"
    "${IBRIDO_PREFIX}/ibrido_logs:/root/ibrido_logs"
    "${IBRIDO_PREFIX}/tmp:/tmp:rw"
    "${IBRIDO_PREFIX}/aux_data:/root/aux_data:rw"
    "${IBRIDO_PREFIX}/training_data:/root/training_data:rw"
    "${IBRIDO_WS_PREFIX}:/root/ibrido_ws:rw"
    "${IBRIDO_CONDA}:/opt/conda:rw"
    "${IBRIDO_PREFIX}/conda_hidden/.conda:/root/.conda:rw"
    "${IBRIDO_PREFIX}/.cache/wandb:/root/.cache/wandb:rw"
    "${IBRIDO_PREFIX}/.byobu:/root/.byobu:rw"
    "${IBRIDO_PREFIX}/.ros:/root/.ros:rw"
    "${IBRIDO_PREFIX}/.rviz2:/root/.rviz2:rw"
    "${IBRIDO_PREFIX}/.mamba:/root/.mamba:rw"
)

# Only add these bindings if PBS is NOT available (when runnin on cluster
# we don't need user input)
if [ "$IS_PBS_AVAILABLE" = false ]; then
    IBRIDO_BDIRS+=(
        "/dev/input:/dev/input:rw"
        "/tmp/.X11-unix:/tmp/.X11-unix"
    )
fi

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
    "git@github.com:AndrePatri/AugMPC.git*ibrido"
    "git@github.com:AndrePatri/AugMPCEnvs.git*ibrido"
    "git@github.com:AndrePatri/MPCHive.git*devel"
    "git@github.com:AndrePatri/EigenIPC.git*devel"
    "git@github.com:AndrePatri/MPCViz.git*ros2_humble"
    "git@github.com:ADVRHumanoids/KyonRLStepping.git*ibrido"
    "git@github.com:ADVRHumanoids/CentauroHybridMPC.git*ibrido"
    "git@github.com:AndrePatri/horizon.git*ibrido"
    "git@github.com:AndrePatri/phase_manager.git*ibrido"
    "git@github.com:AndrePatri/unitree_ros.git*ibrido"
    "git@github.com:AndrePatri/iit-centauro-ros-pkg.git*ibrido_ros2"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*ibrido_ros2_simple"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*ibrido_ros2&iit-kyon-description"
    "git@github.com:AndrePatri/casadi.git*optional_float"
    "git@gitlab.com:crzz/adarl.git*andrepatri_dev"
)

# Concatenate
IBRIDO_BDIRS=("${IBRIDO_BDIRS[@]}" "${ISAAC_BDIRS[@]}")
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

# # extract git repo info
# IBRIDO_GIT_SRC=()
# IBRIDO_GIT_BRCH=()
# for entry in "${IBRIDO_GITDIRS[@]}"; do
#     # Split entry based on colon
#     IFS='*' read -r src branch <<< "$entry"
    
#     # Add to respective arrays
#     IBRIDO_GIT_SRC+=("$src")
#     IBRIDO_GIT_BRCH+=("$branch")
# done

# extract git repo info
IBRIDO_GIT_SRC=()
IBRIDO_GIT_BRCH=()
IBRIDO_GIT_DIR=()
for entry in "${IBRIDO_GITDIRS[@]}"; do
# Split entry on first '*'
IFS='*' read -r src rest <<< "$entry"


branch_and_dir="$rest"
branch="$branch_and_dir"
dir=""


# If a '->' is present in the rest, split into branch and dir
if [[ "$branch_and_dir" == *'&'* ]]; then
IFS='&' read -r branch dir <<< "$branch_and_dir"
fi


# Trim whitespace (in case of accidental spaces)
branch="$(echo -n "$branch" | xargs)"
dir="$(echo -n "$dir" | xargs)"


IBRIDO_GIT_SRC+=("$src")
IBRIDO_GIT_BRCH+=("$branch")
IBRIDO_GIT_DIR+=("$dir")
done
