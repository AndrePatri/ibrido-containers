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

# setup environment
if [ -f /opt/xbot/setup.sh ]; then
    source /opt/xbot/setup.sh
fi

_ibrido_prepend_path LD_LIBRARY_PATH "/opt/ros/${ROS_DISTRO:-jazzy}/lib"
_ibrido_prepend_path LD_LIBRARY_PATH "/opt/xbot/lib"
_ibrido_prepend_path LD_LIBRARY_PATH "${HOME}/ibrido_ws/install/lib"
_ibrido_prepend_path CMAKE_PREFIX_PATH "/opt/xbot"
_ibrido_prepend_path CMAKE_PREFIX_PATH "${HOME}/ibrido_ws/install"
_ibrido_prepend_path PATH "${HOME}/ibrido_ws/install/bin"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3.12/site-packages"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3.11/site-packages"
_ibrido_prepend_path PYTHONPATH "${HOME}/ibrido_ws/install/lib/python3/dist-packages"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/install/lib"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/install/share"
_ibrido_prepend_path ROS_PACKAGE_PATH "${HOME}/ibrido_ws/ros_src"
_ibrido_prepend_path AMENT_PREFIX_PATH "${HOME}/ibrido_ws/install"
_ibrido_prepend_path PKG_CONFIG_PATH "${HOME}/ibrido_ws/install/lib/pkgconfig"

unset -f _ibrido_prepend_path

ibrido_kill_ws() {
    local session_name="${1:-${IBRIDO_BYOBU_WS_NAME:-${BYOBU_WS_NAME:-ibrido_isaac_5x}}}"
    byobu kill-session -t "${session_name}"
}
