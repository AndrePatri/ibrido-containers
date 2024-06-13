#!/bin/bash
set -e # exiting if any cmd fails

IBRIDO_PREFIX=$HOME/docker/ibrido-singularity
WS_SRC=${IBRIDO_PREFIX}/ibrido_ws/src

mkdir -p ${IBRIDO_PREFIX}/training_data
mkdir -p ${IBRIDO_PREFIX}/aux_data
mkdir -p ${IBRIDO_PREFIX}/conda
mkdir -p ${IBRIDO_PREFIX}/conda_hidden/.conda
mkdir -p ${IBRIDO_PREFIX}/.cache/wandb
mkdir -p ${IBRIDO_PREFIX}/network/
touch ${IBRIDO_PREFIX}/network/.netrc
mkdir -p ${IBRIDO_PREFIX}/.byobu

mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/kit
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/ov
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/pip
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/warp
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/glcache
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/cache/computecache
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/logs
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/exts
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/data
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/apps
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/documents
mkdir -p ${IBRIDO_PREFIX}/isaac-sim/local
mkdir -p $WS_SRC

cd $WS_SRC
git clone -b main git@github.com:AndrePatri/IBRIDO.git
git clone -b isaac4.0.0 git@github.com:AndrePatri/LRHControl.git
git clone -b devel git@github.com:AndrePatri/CoClusterBridge.git
git clone -b devel git@github.com:AndrePatri/SharsorIPCpp.git
git clone -b isaac4.0.0  git@github.com:AndrePatri/OmniRoboGym.git
git clone -b ros2_humble git@github.com:AndrePatri/RHCViz.git
git clone -b new_architecture git@github.com:AndrePatri/phase_manager.git
git clone -b andrepatri_dev git@github.com:AndrePatri/unitree_ros.git
git clone -b andrepatri_devel git@github.com:ADVRHumanoids/horizon.git
git clone -b isaac4.0.0 git@github.com:ADVRHumanoids/KyonRLStepping.git
git clone -b isaac4.0.0 git@github.com:ADVRHumanoids/CentauroHybridMPC.git
git clone -b big_wheels_v2.10_optional_find_ros2 git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git
git clone -b optional_find_ros2 git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git
