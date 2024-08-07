#!/bin/bash
set -e # exiting if any cmd fails

echo "--> Setting up workspace..."

WS_BASEDIR=$HOME/ibrido_ws
XBOT2_SETUP=/opt/xbot2/setup.sh

source /usr/local/bin/_activate_current_env.sh # enable mamba for this shell
micromamba activate ${MAMBA_ENV_NAME} # this has to be active to properly install packages
source $XBOT2_SETUP
source /opt/ros/noetic/setup.bash # ros2 setup
source ${WS_BASEDIR}/setup.bash # ros2 setup

# clean ws if already initialized
rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

# build cmake packages
mkdir -p $WS_BASEDIR/build/SharsorIPCpp
cd $WS_BASEDIR/build/SharsorIPCpp
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/SharsorIPCpp/SharsorIPCpp
make -j8 install

mkdir -p $WS_BASEDIR/build/horizon
cd $WS_BASEDIR/build/horizon
cmake -DCMAKE_BUILD_TYPE=Release ../../src/horizon/horizon/cpp
make -j8 install

mkdir -p $WS_BASEDIR/build/phase_manager
cd $WS_BASEDIR/build/phase_manager
cmake -DCMAKE_BUILD_TYPE=Release ../../src/phase_manager/
make -j8 install

mkdir -p $WS_BASEDIR/build/mujoco_cmake
cd $WS_BASEDIR/build/mujoco_cmake
cmake -DCMAKE_BUILD_TYPE=Release ../../src/mujoco_cmake/
make -j8 install

mkdir -p $WS_BASEDIR/build/xbot2_mujoco
cd $WS_BASEDIR/build/xbot2_mujoco
cmake -DCMAKE_BUILD_TYPE=Release ../../src/xbot2_mujoco/
make -j8 install

# pip installations
cd $WS_BASEDIR/src  
pip install -e CoClusterBridge 
pip install -e OmniRoboGym
pip install -e LRHControl
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip install -e RHCViz
pip install --no-deps -e horizon

# copying omnirobogym isaac kit 
# cd $WS_BASEDIR/src/OmniRoboGym/omni_robo_gym/cfg/omni_kits/  
# ./copy2isaac_folder.sh

# copying script to launch byobu
#cp $WS_BASEDIR/src/LRHControl/lrhc_control/scripts/launch_byobu_ws.sh /root/

source /root/.bashrc




