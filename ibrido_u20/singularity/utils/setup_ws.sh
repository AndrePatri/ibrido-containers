#!/bin/bash
set -e # exiting if any cmd fails
echo "--> Setting up workspace..."

WS_BASEDIR=$HOME/ibrido_ws
XBOT2_SETUP=/opt/xbot/setup.sh
WS_INSTALLDIR=$HOME/ibrido_ws/install

source /opt/ros/noetic/setup.bash # ros2 setup
source ${WS_BASEDIR}/setup.bash # ros2 setup

# clean ws if already initialized
rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

# OUTSIDE MICROMAMBA ENV->

# build cmake packages
mkdir -p $WS_BASEDIR/build/mujoco_cmake
cd $WS_BASEDIR/build/mujoco_cmake
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WS_INSTALLDIR} ../../src/mujoco_cmake/
make -j8 install

mkdir -p $WS_BASEDIR/build/xbot2_mujoco
cd $WS_BASEDIR/build/xbot2_mujoco
source $XBOT2_SETUP
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_TESTS=1 -DWITH_XMJ_SIM_ENV=1 -DWITH_PYTHON=1 -DCMAKE_INSTALL_PREFIX=${WS_INSTALLDIR} -Diit_centauro_ros_pkg_DIR=${WS_BASEDIR}/src/iit-centauro-ros-pkg ../../src/xbot2_mujoco/
make -j8 install

# adarl ros utils
mkdir -p $WS_BASEDIR/build/adarl_ros_utils
cd $WS_BASEDIR/build/adarl_ros_utils
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WS_INSTALLDIR} -DWITH_MOVEIT=0 -DWITH_ROS_CONTROL=0 ../../src/adarl_ros/adarl_ros_utils/
make -j8 install

# INSIDE MICROMAMBA ENV->
source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell
micromamba activate ${MAMBA_ENV_NAME} # this has to be active to properly install packages

mkdir -p $WS_BASEDIR/build/perf_sleep
cd $WS_BASEDIR/build/perf_sleep
cmake -DCMAKE_BUILD_TYPE=Release ../../src/PerfSleep/perf_sleep
make -j8 install

mkdir -p $WS_BASEDIR/build/SharsorIPCpp
cd $WS_BASEDIR/build/SharsorIPCpp
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/SharsorIPCpp/SharsorIPCpp
make -j8 install

mkdir -p $WS_BASEDIR/build/casadi
# cd $WS_BASEDIR/build/casadi
# cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OSQP=1 -DWITH_QPOASES=1 -DWITH_LAPACK=1 -DWITH_THREAD=1 -DWITH_PYTHON=1 -DWITH_PYTHON3=1 -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/casadi
# make -j8 install

mkdir -p $WS_BASEDIR/build/horizon
cd $WS_BASEDIR/build/horizon
cmake -DCMAKE_BUILD_TYPE=Release ../../src/horizon/horizon/cpp
make -j8 install

mkdir -p $WS_BASEDIR/build/phase_manager
cd $WS_BASEDIR/build/phase_manager
cmake -DCMAKE_BUILD_TYPE=Release ../../src/phase_manager/
make -j8 install

pip install -e $WS_BASEDIR/src/jumping_leg

# pip installations
cd $WS_BASEDIR/src  
pip install -e CoClusterBridge 
pip install -e LRHControlEnvs
pip install -e LRHControl
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip install -e RHCViz
# pip install --no-deps -e horizon --install-option="--skip-build"
pip install --no-deps -e horizon
pip install -e adarl
pip install -e adarl_ros/adarl_ros

# copying lrhcontrolenvs isaac kit 
# cd $WS_BASEDIR/src/LRHControlEnvs/lrhcontrolenvs/cfg/omni_kits/  
# ./copy2isaac_folder.sh

# copying script to launch byobu
#cp $WS_BASEDIR/src/LRHControl/lrhc_control/scripts/launch_byobu_ws.sh /root/

source /root/.bashrc