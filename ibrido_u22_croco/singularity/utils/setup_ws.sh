#!/bin/bash
set -e # exiting if any cmd fails

echo "--> Setting up workspace..."

WS_BASEDIR=$HOME/ibrido_ws

source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell
source /opt/ros/humble/setup.bash # ros2 setup
source $WS_BASEDIR/setup.bash

echo -e "\nWill use cmake version:" 
cmake --version

echo -e "\ngcc version:" 
g++ --version

# clean ws if already initialized
rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

# mkdir -p $WS_BASEDIR/build/cppad
# cd $WS_BASEDIR/build/cppad
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/CppAD
# make -j8 install

micromamba activate ${MAMBA_ENV_NAME} # this has to be active to properly install some packages

# mkdir -p $WS_BASEDIR/build/cppadcg
# cd $WS_BASEDIR/build/cppadcg
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/CppADCodeGen
# make -j8 install

# build cmake packages
mkdir -p $WS_BASEDIR/build/EigenIPC
cd $WS_BASEDIR/build/EigenIPC
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/EigenIPC/EigenIPC
make -j8 install

# # mkdir -p $WS_BASEDIR/build/casadi
# # cd $WS_BASEDIR/build/casadi
# # cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OSQP=1 -DWITH_QPOASES=1 -DWITH_LAPACK=1 -DWITH_THREAD=1 -DWITH_PYTHON=1 -DWITH_PYTHON3=1 -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/casadi
# # make -j8 install

# mkdir -p $WS_BASEDIR/build/horizon
# cd $WS_BASEDIR/build/horizon
# cmake -DCMAKE_BUILD_TYPE=Release ../../src/horizon/horizon/cpp
# make -j8 install

# mkdir -p $WS_BASEDIR/build/phase_manager
# cd $WS_BASEDIR/build/phase_manager
# cmake -DCMAKE_BUILD_TYPE=Release ../../src/phase_manager/
# make -j8 install

mkdir -p $WS_BASEDIR/build/centauro_urdf
cd $WS_BASEDIR/build/centauro_urdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/iit-centauro-ros-pkg/centauro_urdf
make -j8 install

mkdir -p $WS_BASEDIR/build/kyon_urdf
cd $WS_BASEDIR/build/kyon_urdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/iit-kyon-ros-pkg/kyon_urdf
make -j8 install

mkdir -p $WS_BASEDIR/build/kyon_srdf
cd $WS_BASEDIR/build/kyon_srdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/iit-kyon-ros-pkg/kyon_srdf
make -j8 install

echo -e "\nCompiling example-robot-data" 
mkdir -p $WS_BASEDIR/build/example-robot-data
cd $WS_BASEDIR/build/example-robot-data
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/example-robot-data
make -j8 install

# pip installations
cd $WS_BASEDIR/src  
pip install -e CoClusterBridge 
pip install -e LRHControlEnvs
pip install -e LRHControl
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip install -e RHCViz
# pip install -e adarl

# pip install --no-deps -e horizon

echo -e "\nCompiling crocoddyl" 
mkdir -p $WS_BASEDIR/build/crocoddyl
cd $WS_BASEDIR/build/crocoddyl
# CC=clang CXX=clang++ 
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" \
  -DBUILD_WITH_CODEGEN_SUPPORT=1 \
  -DBUILD_WITH_MULTITHREADS=1 \
  ../../src/crocoddyl
make -j5 install

source /root/.bashrc

echo 'setup completed.'





