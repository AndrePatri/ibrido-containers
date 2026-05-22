#!/bin/bash
set -e # exiting if any cmd fails

echo "--> Setting up workspace..."

WS_BASEDIR=$HOME/ibrido_ws
MAMBA_ENV_NAME_ISAAC="${MAMBA_ENV_NAME_ISAAC:-${MAMBA_ENV_NAME}_isaac_py11}"

build_cmake_pkg_if_present() {
    local label="$1"
    local source_dir="$2"
    shift 2

    if [ ! -d "$WS_BASEDIR/src/$source_dir" ]; then
        echo "--> Skipping $label: $WS_BASEDIR/src/$source_dir not found."
        return 0
    fi

    mkdir -p "$WS_BASEDIR/build/$label"
    cd "$WS_BASEDIR/build/$label"
    cmake "$@" "../../src/$source_dir"
    make -j8 install
}

pip_install_if_present() {
    local package_dir="$1"

    if [ ! -d "$WS_BASEDIR/src/$package_dir" ]; then
        echo "--> Skipping pip install for $package_dir: directory not found."
        return 0
    fi

    pip install -e "$package_dir"
}

# clean ws if already initialized
rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell

# due to Isaac 5.1 only supporting python 3.11, we use two separate mamba envs, one for isaac and one for the rest
# which can also be used with prebuilt ros2 jazzy packages. This means we need to install the base ibrido packages
# in both envs.
micromamba activate ${MAMBA_ENV_NAME_ISAAC}
export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME_ISAAC/lib:$LD_LIBRARY_PATH

mkdir -p $WS_BASEDIR/build/perf_sleep
cd $WS_BASEDIR/build/perf_sleep
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/PerfSleep/perf_sleep
make -j8 install

mkdir -p $WS_BASEDIR/build/EigenIPC
cd $WS_BASEDIR/build/EigenIPC
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/EigenIPC/EigenIPC
make -j8 install

# pip installations
cd $WS_BASEDIR/src
pip install -e MPCHive
pip install -e AugMPCEnvs
pip install -e AugMPC

# clean cmake cache and builds
rm -rf $WS_BASEDIR/build/perf_sleep && mkdir $WS_BASEDIR/build/perf_sleep
rm -rf $WS_BASEDIR/build/EigenIPC && mkdir $WS_BASEDIR/build/EigenIPC

micromamba deactivate

micromamba activate ${MAMBA_ENV_NAME} # this has to be active to properly install packages
export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME/lib:$LD_LIBRARY_PATH

# build cmake packages

mkdir -p $WS_BASEDIR/build/gtest
cd $WS_BASEDIR/build/gtest
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/googletest
make -j4 install

mkdir -p $WS_BASEDIR/build/perf_sleep
cd $WS_BASEDIR/build/perf_sleep
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/PerfSleep/perf_sleep
make -j8 install

mkdir -p $WS_BASEDIR/build/EigenIPC
cd $WS_BASEDIR/build/EigenIPC
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/EigenIPC/EigenIPC
make -j8 install

mkdir -p $WS_BASEDIR/build/phase_manager
cd $WS_BASEDIR/build/phase_manager
cmake -DCMAKE_BUILD_TYPE=Release -DTESTS=OFF ../../src/phase_manager/
make -j8 install

mkdir -p $WS_BASEDIR/build/casadi_kin_dyn
cd $WS_BASEDIR/build/casadi_kin_dyn
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/casadi_kin_dyn/
make -j8 install

# mkdir -p $WS_BASEDIR/build/casadi
# cd $WS_BASEDIR/build/casadi
# cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OSQP=1 -DWITH_QPOASES=1 -DWITH_LAPACK=1 -DWITH_THREAD=1 -DWITH_PYTHON=1 -DWITH_PYTHON3=1 -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/casadi
# make -j8 install

mkdir -p $WS_BASEDIR/build/horizon
cd $WS_BASEDIR/build/horizon
cmake -DCMAKE_BUILD_TYPE=Release -DTESTS=OFF ../../src/horizon/horizon/cpp
make -j8 install

source /opt/ros/jazzy/setup.bash # ros2 setup and ros2 dependent-packages installation

mkdir -p $WS_BASEDIR/build/centauro_urdf
cd $WS_BASEDIR/build/centauro_urdf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/iit-centauro-ros-pkg/centauro_urdf
make -j8 install

build_cmake_pkg_if_present kyon_urdf_simple iit-kyon-ros-pkg/kyon_urdf \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

build_cmake_pkg_if_present kyon_srdf_simple iit-kyon-ros-pkg/kyon_srdf \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

build_cmake_pkg_if_present kyon_urdf iit-kyon-description/kyon_urdf \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

build_cmake_pkg_if_present kyon_srdf iit-kyon-description/kyon_srdf \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

# pip installations into mamba env
cd $WS_BASEDIR/src
pip install -e MPCHive
pip install -e AugMPCEnvs
pip install -e AugMPC
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip_install_if_present TalosHybridMPC
pip install -e MPCViz
pip install -e adarl
pip install --no-deps -e horizon
micromamba install -y clang

source /root/.bashrc

echo 'setup completed.'
