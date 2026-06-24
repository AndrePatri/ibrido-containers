#!/bin/bash
set -e # exiting if any cmd fails

echo "--> Setting up workspace..."

WS_BASEDIR=$HOME/ibrido_ws
MAMBA_ENV_NAME_ISAAC="${MAMBA_ENV_NAME_ISAAC:-${MAMBA_ENV_NAME}_isaac_py11}"
XBOT2_SETUP="${XBOT2_SETUP:-/opt/xbot/setup.sh}"
XBOT2_INSTALL_PREFIX="${XBOT2_INSTALL_PREFIX:-$WS_BASEDIR/install}"
SYSTEM_ENV_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MAIN_ENV_PYTHON="$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME/bin/python"

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

install_mamba_runtime_hook() {
    local env_name="$1"
    local env_prefix="$MAMBA_ROOT_PREFIX/envs/$env_name"
    local hook_dir="$env_prefix/etc/conda/activate.d"

    if [ ! -d "$env_prefix" ]; then
        echo "--> Cannot install runtime hook: mamba env not found at $env_prefix"
        exit 1
    fi

    mkdir -p "$hook_dir"
    cat > "$hook_dir/ibrido_runtime_paths.sh" <<'EOF'
_ibrido_prepend_path() {
    local var_name="$1"
    local new_path="$2"
    local current_value="${!var_name:-}"

    [ -d "$new_path" ] || return 0
    case ":${current_value}:" in
        *":${new_path}:"*) ;;
        *) export "${var_name}=${new_path}${current_value:+:${current_value}}" ;;
    esac
}

_ibrido_prepend_path LD_LIBRARY_PATH "/opt/ros/${ROS_DISTRO:-jazzy}/lib"
_ibrido_prepend_path LD_LIBRARY_PATH "/opt/xbot/lib"
_ibrido_prepend_path LD_LIBRARY_PATH "${HOME}/ibrido_ws/install/lib"
_ibrido_prepend_path CMAKE_PREFIX_PATH "/opt/xbot"
_ibrido_prepend_path CMAKE_PREFIX_PATH "${HOME}/ibrido_ws/install"
_ibrido_prepend_path PATH "${HOME}/ibrido_ws/install/bin"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3.12/site-packages"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3.11/site-packages"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3/dist-packages"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/ros_src"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/install/share"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/install/lib"
_ibrido_prepend_path AMENT_PREFIX_PATH "${HOME}/ibrido_ws/install"
_ibrido_prepend_path PKG_CONFIG_PATH "${HOME}/ibrido_ws/install/lib/pkgconfig"

unset -f _ibrido_prepend_path
EOF
}

resolve_xbot2_setup() {
    local xbot2_setup="$XBOT2_SETUP"

    if [ ! -f "$XBOT2_SETUP" ]; then
        local detected_xbot_setup
        detected_xbot_setup="$(find /opt /usr/local /usr -path '*xbot*setup.sh' -print -quit 2>/dev/null || true)"
        if [ -n "$detected_xbot_setup" ]; then
            xbot2_setup="$detected_xbot_setup"
        fi
    fi

    if [ ! -f "$xbot2_setup" ]; then
        echo "--> Cannot build XBot2 packages: XBot2 setup file not found."
        echo "--> Looked at /opt/xbot/setup.sh and under /opt, /usr/local, /usr."
        echo "--> Rebuild the u24 image with XBot2 nightly support, or install/build XBot2 in the image first."
        exit 1
    fi

    RESOLVED_XBOT2_SETUP="$xbot2_setup"
}

setup_system_xbot_build_env() {
    local xbot2_setup="$1"

    unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_LIBRARY_PATH PYTHONPATH
    unset CMAKE_PREFIX_PATH PKG_CONFIG_PATH CONDA_PREFIX
    export PATH="$SYSTEM_ENV_PATH"

    source /opt/ros/jazzy/setup.bash
    source "$xbot2_setup"

    export LD_LIBRARY_PATH="$WS_BASEDIR/install/lib:${LD_LIBRARY_PATH:-}"
    export CMAKE_PREFIX_PATH="$WS_BASEDIR/install:${CMAKE_PREFIX_PATH:-}"
    export PATH="$WS_BASEDIR/install/bin:$PATH"
}

install_mujoco_cmake() {
    (
        setup_system_xbot_build_env "$1"

        mkdir -p "$WS_BASEDIR/build/mujoco_cmake"
        cd "$WS_BASEDIR/build/mujoco_cmake"
        /usr/bin/cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$WS_BASEDIR/install" \
            "$WS_BASEDIR/src/mujoco_cmake"
        /usr/bin/make -j8 install
    )
}

install_xbot2_mujoco() {
    (
        setup_system_xbot_build_env "$1"

        mkdir -p "$WS_BASEDIR/build/xbot2_mujoco"
        cd "$WS_BASEDIR/build/xbot2_mujoco"
        /usr/bin/cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DWITH_TESTS=0 \
            -DWITH_XMJ_SIM_ENV=1 \
            -DWITH_PYTHON=1 \
            -DCMAKE_INSTALL_PREFIX="$XBOT2_INSTALL_PREFIX" \
            -DPYTHON_EXECUTABLE="$MAIN_ENV_PYTHON" \
            -DPython3_EXECUTABLE="$MAIN_ENV_PYTHON" \
            -Diit_centauro_ros_pkg_DIR="$WS_BASEDIR/src/iit-centauro-ros-pkg" \
            "$WS_BASEDIR/src/xbot2_mujoco"
        /usr/bin/make -j8 install
    )
}

install_xbot2_zmq() {
    (
        setup_system_xbot_build_env "$1"

        mkdir -p "$WS_BASEDIR/build/xbot2_zmq"
        cd "$WS_BASEDIR/build/xbot2_zmq"
        /usr/bin/cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$XBOT2_INSTALL_PREFIX" \
            "$WS_BASEDIR/src/xbot2_zmq"
        /usr/bin/make -j8 install
    )
}

# clean ws if already initialized
rm -rf $WS_BASEDIR/build && mkdir $WS_BASEDIR/build
rm -rf $WS_BASEDIR/install && mkdir $WS_BASEDIR/install

source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell

install_mamba_runtime_hook "$MAMBA_ENV_NAME"
install_mamba_runtime_hook "$MAMBA_ENV_NAME_ISAAC"

# due to Isaac 5.1 only supporting python 3.11, we use two separate mamba envs, one for isaac and one for the rest
# which can also be used with prebuilt ros2 jazzy packages. This means we need to install the base ibrido packages
# in both envs.
micromamba activate ${MAMBA_ENV_NAME_ISAAC}
export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME_ISAAC/lib:$LD_LIBRARY_PATH

mkdir -p $WS_BASEDIR/build/EigenIPC
cd $WS_BASEDIR/build/EigenIPC
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON ../../src/EigenIPC/EigenIPC
make -j8 install

# pip installations
cd $WS_BASEDIR/src
pip install --no-deps -e xbot2_zmq/pyxbot
pip install -e MPCHive
pip install -e AugMPCEnvs
pip install -e AugMPC

# clean cmake cache and builds
rm -rf $WS_BASEDIR/build/EigenIPC && mkdir $WS_BASEDIR/build/EigenIPC

micromamba deactivate

micromamba activate ${MAMBA_ENV_NAME} # this has to be active to properly install packages
export LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME/lib:$LD_LIBRARY_PATH

# build cmake packages

mkdir -p $WS_BASEDIR/build/gtest
cd $WS_BASEDIR/build/gtest
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install" ../../src/googletest
make -j4 install

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

# build_cmake_pkg_if_present kyon_urdf_simple iit-kyon-ros-pkg/kyon_urdf \
#     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

# build_cmake_pkg_if_present kyon_srdf_simple iit-kyon-ros-pkg/kyon_srdf \
#     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

# build_cmake_pkg_if_present kyon_urdf iit-kyon-description/kyon_urdf \
#     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

# build_cmake_pkg_if_present kyon_srdf iit-kyon-description/kyon_srdf \
#     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/ibrido_ws/install"

resolve_xbot2_setup
install_mujoco_cmake "$RESOLVED_XBOT2_SETUP"
install_xbot2_mujoco "$RESOLVED_XBOT2_SETUP"
install_xbot2_zmq "$RESOLVED_XBOT2_SETUP"

# pip installations into mamba env
cd $WS_BASEDIR/src
pip install --no-deps -e xbot2_zmq/pyxbot
pip install -e MPCHive
pip install -e AugMPCEnvs
pip install -e AugMPC
pip install -e CentauroHybridMPC
pip install -e KyonRLStepping
pip_install_if_present TalosHybridMPC
pip install -e MPCViz
pip install -e adarl
pip install --no-deps -e horizon

pip install -e genesis-world[dev]

micromamba install -y clang

source /root/.bashrc

echo 'setup completed.'
