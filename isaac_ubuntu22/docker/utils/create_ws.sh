#!/bin/bash
set -e # exiting if any cmd fails

WS_SRC=$HOME/docker/ibrido-docker/ibrido_ws/src

mkdir -p $HOME/docker/ibrido-docker/aux_data
mkdir -p $HOME/docker/ibrido-docker/conda
mkdir -p $HOME/docker/ibrido-docker/isaac-sim
mkdir -p $WS_SRC

cd $WS_SRC
git clone -b main git@github.com:AndrePatri/IBRIDO.git
git clone -b isaac2023.1.1_dev git@github.com:AndrePatri/LRHControl.git
git clone -b devel git@github.com:AndrePatri/CoClusterBridge.git
git clone -b devel git@github.com:AndrePatri/SharsorIPCpp.git
git clone -b isaac2023.1.1_dev git@github.com:AndrePatri/OmniRoboGym.git
git clone -b ros2_humble git@github.com:AndrePatri/RHCViz.git
git clone -b new_architecture git@github.com:AndrePatri/phase_manager.git
git clone -b andrepatri_dev git@github.com:AndrePatri/unitree_ros.git
git clone -b andrepatri_devel git@github.com:ADVRHumanoids/horizon.git
git clone -b isaac2023.1.1_dev git@github.com:ADVRHumanoids/KyonRLStepping.git
git clone -b isaac2023.1.1_dev git@github.com:ADVRHumanoids/CentauroHybridMPC.git
git clone -b big_wheels_v2.10_optional_find_ros2 git@github.com:ADVRHumanoids/iit-centauro-ros-pkg.git
git clone -b optional_find_ros2 git@github.com:ADVRHumanoids/iit-kyon-ros-pkg.git
