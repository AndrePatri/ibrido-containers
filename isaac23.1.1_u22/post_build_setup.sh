#!/bin/bash

echo "Running post-build steps. It may take a while...."

# ws installation
# source /opt/ros/humble/setup.bash
# WS_BASEDIR=$HOME/RL_ws/hhcm
# WS_INSTALLDIR=$HOME/RL_ws/hhcm

# cd $WS_BASEDIR/build/SharsorIPCpp
# cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/SharsorIPCpp/SharsorIPCpp
# make install 
# cd $WS_BASEDIR/build/centauro_urdf
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$WS_INSTALLDIR ../../iit-centauro-ros-pkg/centauro_urdf
# make install

# Launch a shell session
/bin/bash