#!/bin/bash

WS_BASEDIR=$HOME/RL_ws/hhcm

rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

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

mkdir -p $WS_BASEDIR/build/centauro_urdf
cd $WS_BASEDIR/build/centauro_urdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/RL_ws/hhcm/install" ../../src/iit-centauro-ros-pkg/centauro_urdf
make -j8 install

mkdir -p $WS_BASEDIR/build/kyon_urdf
cd $WS_BASEDIR/build/kyon_urdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/RL_ws/hhcm/install" ../../src/iit-kyon-ros-pkg/kyon_urdf
make -j8 install

mkdir -p $WS_BASEDIR/build/kyon_srdf
cd $WS_BASEDIR/build/kyon_srdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/RL_ws/hhcm/install" ../../src/iit-kyon-ros-pkg/kyon_srdf
make -j8 install

cd $WS_BASEDIR/src  

pip install -e CoClusterBridge 
pip install -e OmniRoboGym
pip install -e LRHControl
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip install -e RHCViz
pip install --no-deps -e horizon

cd $WS_BASEDIR/src/OmniRoboGym/omni_robo_gym/cfg/omni_kits/  
./copy2isaac_folder.sh




