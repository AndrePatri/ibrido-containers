#!/bin/bash
set -e # exiting if any cmd fails

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_folder="$(dirname "$THIS_DIR")"

source "${root_folder}/files/bind_list.sh"

echo 'Creating all directories...'
mkdir -p $IBRIDO_WS_SRC # to hold git repos
for item in "${IBRIDO_BDIRS_SRC[@]}"; do
    echo "--> $item"
    mkdir -p $item
done

echo 'Cloning repos...'
cd $IBRIDO_WS_SRC
for ((i = 0; i < ${#IBRIDO_GITDIRS[@]}; i++)); do
    src="${IBRIDO_GIT_SRC[$i]}"
    branch="${IBRIDO_GIT_BRCH[$i]}"
    echo "--> $src # $branch"
    git clone -q -b $branch $src
done
# echo 'Cloning all repos into workspace (if not already there)...'
# git clone -q -b main git@github.com:AndrePatri/IBRIDO.git &
# git clone -q -b isaac4.0.0 git@github.com:AndrePatri/LRHControl.git &
# git clone -q -b devel git@github.com:AndrePatri/CoClusterBridge.git &
# git clone -q -b devel git@github.com:AndrePatri/SharsorIPCpp.git &
# git clone -q -b isaac4.0.0  git@github.com:AndrePatri/OmniRoboGym.git &
# git clone -q -b ros2_humble git@github.com:AndrePatri/RHCViz.git &
# git clone -q -b new_architecture git@github.com:AndrePatri/phase_manager.git &
# git clone -q -b andrepatri_dev git@github.com:AndrePatri/unitree_ros.git &
# git clone -q -b andrepatri_devel git@github.com:ADVRHumanoids/horizon.git &
# git clone -q -b isaac4.0.0 git@github.com:ADVRHumanoids/KyonRLStepping.git &
# git clone -q -b isaac4.0.0 git@github.com:ADVRHumanoids/CentauroHybridMPC.git &
# git clone -q -b big_wheels_v2.10_optional_find_ros2 git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git &
# git clone -q -b optional_find_ros2 git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git &
# wait
# echo 'Done.'
