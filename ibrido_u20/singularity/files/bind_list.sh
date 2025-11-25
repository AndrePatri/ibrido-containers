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
    "/run/user/1000:/run/user/1000"
    # "/media/apatrizi/AP_ext/data/IBRIDO/:/root/IBRIDO"
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
    "${IBRIDO_PREFIX}/.mamba:/root/.mamba:rw"
    "${IBRIDO_PREFIX}/.config:/root/.config:rw"
)

OTHER_BDIRS=(
    ":"
)

# git directories and their branches
IBRIDO_GITDIRS=(
    "git@github.com:AndrePatri/IBRIDO.git*main"
    "git@github.com:AndrePatri/AugMPC.git*ibrido"
    "git@github.com:AndrePatri/AugMPCEnvs.git*ibrido"
    "git@github.com:AndrePatri/MPCHive.git*devel"
    "git@github.com:AndrePatri/EigenIPC.git*devel"
    "git@github.com:AndrePatri/MPCViz.git*ros1_noetic"
    "git@github.com:ADVRHumanoids/KyonRLStepping.git*ibrido"
    "git@github.com:ADVRHumanoids/CentauroHybridMPC.git*ibrido"
    "git@github.com:AndrePatri/horizon.git*ibrido"
    "git@github.com:AndrePatri/phase_manager.git*ibrido"
    "git@github.com:AndrePatri/xbot2_mujoco.git*ibrido"
    "git@github.com:AndrePatri/mujoco_cmake.git*ibrido"
    "git@github.com:AndrePatri/unitree_ros.git*ibrido"
    "git@github.com:AndrePatri/iit-centauro-ros-pkg.git*ibrido_ros1"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*ibrido_ros1_simple"
    "git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git*ibrido_ros1&iit-kyon-description"
    "git@github.com:AndrePatri/PerfSleep.git*main"
    "git@github.com:AndrePatri/casadi.git*optional_float"
    "git@gitlab.com:crzz/adarl.git*ibrido"
    "git@gitlab.com:crzz/adarl_ros.git*ibrido"
    "git@github.com:google/googletest.git*main"
    "git@github.com:ADVRHumanoids/robot_monitoring.git*v2.7.5"
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

# Decide whether to convert git@ssh URLs to HTTPS
# Accepts IBRIDO_USE_HTTP=true|1 (case insensitive) to enable conversion.
USE_HTTP_RAW="${IBRIDO_USE_HTTP:-}"
USE_HTTP="$(echo -n "$USE_HTTP_RAW" | tr '[:upper:]' '[:lower:]')"
if [ "$USE_HTTP" = "true" ] || [ "$USE_HTTP" = "1" ]; then
    CONVERT_TO_HTTP=true
else
    CONVERT_TO_HTTP=false
fi

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

    # If a '&' is present in the rest, split into branch and dir
    if [[ "$branch_and_dir" == *'&'* ]]; then
        IFS='&' read -r branch dir <<< "$branch_and_dir"
    fi

    # Trim whitespace (in case of accidental spaces)
    branch="$(echo -n "$branch" | xargs)"
    dir="$(echo -n "$dir" | xargs)"

    # Optionally convert SSH 'git@host:owner/repo.git' style to HTTPS
    if [ "$CONVERT_TO_HTTP" = true ]; then
        converted="$src"
        case "$src" in
            git@github.com:*)
                converted="https://github.com/${src#git@github.com:}"
                ;;
            git@gitlab.com:*)
                converted="https://gitlab.com/${src#git@gitlab.com:}"
                ;;
            git@bitbucket.org:*)
                converted="https://bitbucket.org/${src#git@bitbucket.org:}"
                ;;
            ssh://git@*)
                # e.g. ssh://git@github.com/owner/repo.git -> https://github.com/owner/repo.git
                # remove "ssh://git@" and prepend https://
                host_and_path="${src#ssh://git@}"
                converted="https://${host_and_path}"
                ;;
            # if it's already https or http, keep it as-is
            http://*|https://*)
                converted="$src"
                ;;
            *)
                # unknown pattern: leave as-is
                converted="$src"
                ;;
        esac
        # assign the converted value
        src="$converted"
    fi

    IBRIDO_GIT_SRC+=("$src")
    IBRIDO_GIT_BRCH+=("$branch")
    IBRIDO_GIT_DIR+=("$dir")
done