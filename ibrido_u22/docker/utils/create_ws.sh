#!/bin/bash
set -e # exiting if any cmd fails

WS_SRC=$HOME/docker/ibrido-docker/ibrido_ws/src

mkdir -p $HOME/docker/ibrido-docker/aux_data
mkdir -p $HOME/docker/ibrido-docker/conda
mkdir -p $HOME/docker/ibrido-docker/isaac-sim
mkdir -p $WS_SRC

cd $WS_SRC
git clone -q -b main git@github.com:AndrePatri/IBRIDO.git &
git clone -q -b isaac4.0.0 git@github.com:AndrePatri/AugMPC.git &
git clone -q -b devel git@github.com:AndrePatri/MPCHive.git &
git clone -q -b devel git@github.com:AndrePatri/EigenIPC.git &
git clone -q -b isaac4.0.0  git@github.com:AndrePatri/AugMPCEnvs.git &
git clone -q -b ros2_humble git@github.com:AndrePatri/MPCViz.git &
git clone -q -b new_architecture git@github.com:AndrePatri/phase_manager.git &
git clone -q -b andrepatri_dev git@github.com:AndrePatri/unitree_ros.git &
git clone -q -b andrepatri_devel git@github.com:ADVRHumanoids/horizon.git &
git clone -q -b isaac4.0.0 git@github.com:ADVRHumanoids/KyonRLStepping.git &
git clone -q -b isaac4.0.0 git@github.com:ADVRHumanoids/CentauroHybridMPC.git &
git clone -q -b big_wheels_v2.10_optional_find_ros2 git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git &
git clone -q -b optional_find_ros2 git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git &

wait
