#!/bin/bash

source /usr/local/bin/_activate_current_env.sh # enable mamba for this shell
micromamba activate LRHControlMambaEnv # this has to be active to properly install packages
source /opt/ros/humble/setup.bash # ros2 setup

WS_BASEDIR=$HOME/RL_ws/hhcm

cp /root/setup.bash $WS_BASEDIR/

clean ws if already initialized
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
cd $WS_BASEDIR/src/OmniRoboGym/omni_robo_gym/cfg/omni_kits/  
./copy2isaac_folder.sh

# copying script to launch byobu
cp $WS_BASEDIR/src/LRHControl/lrhc_control/scripts/launch_byobu_ws.sh /root/

# show git branches from terminal
echo 'parse_git_branch() {' >> /root/.bashrc && \
echo '    git branch 2> /dev/null | sed -e "/^[^*]/d" -e "s/* \(.*\)/ (\1)/"' >> /root/.bashrc && \
echo '}' >> /root/.bashrc && \
echo 'export PS1="\u@\h \[\033[32m\]\w\[\033[33m\]$(parse_git_branch)\[\033[00m\] $ "' >> /root/.bashrc

wandb login --relogin # login to wandb

source /root/.bashrc




