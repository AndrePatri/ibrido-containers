#!/bin/bash

WS_BASEDIR=$HOME/RL_ws/hhcm
cd $WS_BASEDIR/build/SharsorIPCpp
make install
cd $WS_BASEDIR/build/centauro_urdf
make install
cd $WS_BASEDIR/build/horizon
make install
cd $WS_BASEDIR/build/kyon_srdf
make install
cd $WS_BASEDIR/build/kyon_urdf
make install
cd $WS_BASEDIR/build/phase_manager
make install
cd $WS_BASEDIR/build/perf_sleep
make install

cd $WS_BASEDIR/src  

pip install -e CoClusterBridge 
pip install -e OmniRoboGym
pip install -e LRHControl
pip install -e CentauroHybridMPC
pip install -e RHCViz

pip install --no-deps -e horizon

cd $WS_BASEDIR/src/OmniRoboGym/omni_robo_gym/cfg/omni_kits/  
./copy2isaac_folder.sh




