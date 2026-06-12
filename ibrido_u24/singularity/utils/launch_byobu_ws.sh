#!/bin/bash

usage() {
  echo "Usage: $0
    --cfg CFG \
    [--set VAR=VALUE] \
    [--dry-run] \
    "
  exit 1
}

# Set Italian keyboard layout
export XMODIFIERS=@im=ibus
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus

CONTAINER_HOME="${IBRIDO_CONTAINER_HOME:-/root}"
if [ -d "${CONTAINER_HOME}/ibrido_ws" ]; then
    export HOME="${CONTAINER_HOME}"
fi

export BYOBU_CONFIG_DIR="${HOME}/.byobu"
export BYOBU_RUN_DIR="/tmp/byobu-${USER:-ibrido}"
mkdir -p "${BYOBU_CONFIG_DIR}/bin" "${BYOBU_RUN_DIR}"
export PATH="${BYOBU_CONFIG_DIR}/bin:${PATH}"

SLEEP_FOR=0.02
export IBRIDO_BYOBU_WS_NAME="${IBRIDO_BYOBU_WS_NAME:-ibrido_isaac_5x}"
export BYOBU_WS_NAME="${IBRIDO_BYOBU_WS_NAME}"
WS_ROOT="$HOME/ibrido_ws"
DATA_ROOT="$HOME/training_data"
WORKING_DIR="$WS_ROOT/src/AugMPC/aug_mpc/scripts"
WORKING_DIR_QUAD="$WS_ROOT/src/KyonRLStepping/kyonrlstepping/scripts"
WORKING_DIR_CENTAURO="$WS_ROOT/src/CentauroHybridMPC/centaurohybridmpc/scripts"
WORKING_DIR_TALOS="$WS_ROOT/src/TalosHybridMPC/taloshybridmpc/scripts"
WORKING_DIR_JOY="$WS_ROOT/src/MPCHive/mpc_hive/utilities/joy"
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${UTILS_DIR}/ibrido_command_builder.sh"

MAMBAENVNAME="${MAMBA_ENV_NAME}"
MAMBAENVNAME_ISAAC="${MAMBA_ENV_NAME_ISAAC}"

cfg_file_basepath="${IBRIDO_CFG_BASEPATH:-$HOME/ibrido_files/training_cfgs}"
config_file=""
dry_run=0
cfg_overrides=()

custom_arg_value() {
    local arg_name="$1"
    local names=()
    local vals=()
    local i

    read -r -a names <<< "${CUSTOM_ARGS_NAMES:-}"
    read -r -a vals <<< "${CUSTOM_ARGS_VALS:-}"
    for i in "${!names[@]}"; do
        if [ "${names[$i]}" = "$arg_name" ]; then
            printf '%s\n' "${vals[$i]:-}"
            return 0
        fi
    done
    return 1
}

press_enter() {
    byobu send-keys Enter
    sleep $SLEEP_FOR
}

execute_command() {
    byobu send-keys "$1"
    press_enter
}

prepare_command() {
    local cmd="$1"

    cmd="${cmd#reset && }"
    clear_terminal
    byobu send-keys "$cmd"
    sleep $SLEEP_FOR
}

go_to_pane() {
    byobu select-pane -t "$1"
}

current_pane_id() {
    byobu display-message -p '#{pane_id}'
}

attach_to_session() {
    byobu attach-session -t "${BYOBU_WS_NAME}"
}

activate_mamba_env() {
    execute_command "eval \"\$(micromamba shell hook --shell bash)\""
    execute_command "micromamba activate ${MAMBAENVNAME}"
}

activate_mamba_env_isaac() {
    execute_command "eval \"\$(micromamba shell hook --shell bash)\""
    execute_command "micromamba activate ${MAMBAENVNAME_ISAAC}"
}

clear_terminal() {
    byobu send-keys C-l
    sleep $SLEEP_FOR
}

increase_file_limits_locally() {
    execute_command "ulimit -n ${N_FILES}"
}

split_h() {
    byobu split-window -v -l 50%
}

split_v() {
    byobu split-window -h -l 50%
}

new_tab() {
    byobu new-window -n "$1"
}

enabled() {
    case "${1:-0}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

mpcviz_custom_xacro_args_for() {
    local family

    IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS=""
    if [ -z "${CUSTOM_ARGS_NAMES:-}" ]; then
        return
    fi

    for family in "$@"; do
        if [ "${ROBOT_FAMILY:-}" = "$family" ]; then
            IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS=" --custom_args_names $CUSTOM_ARGS_NAMES --custom_args_dtype $CUSTOM_ARGS_DTYPE --custom_args_vals $CUSTOM_ARGS_VALS"
            return
        fi
    done
}

install_byobu_helper_commands() {
    local kill_helper="${BYOBU_CONFIG_DIR}/bin/ibrido_kill_ws"

    cat > "${kill_helper}" <<'EOF'
#!/usr/bin/env bash
set -e

session_name="${1:-${IBRIDO_BYOBU_WS_NAME:-${BYOBU_WS_NAME:-ibrido_isaac_5x}}}"
exec byobu kill-session -t "${session_name}"
EOF
    chmod +x "${kill_helper}"
}

install_terminal_helpers() {
    execute_command "export IBRIDO_BYOBU_WS_NAME=\"${IBRIDO_BYOBU_WS_NAME}\"; export BYOBU_WS_NAME=\"${BYOBU_WS_NAME}\"; export PATH=\"${BYOBU_CONFIG_DIR}/bin:\${PATH}\""
}

setup_main_env_pane() {
    local workdir="$1"
    local setup_cmd
    local extra_cmd
    shift

    setup_cmd="export IBRIDO_BYOBU_WS_NAME=\"${IBRIDO_BYOBU_WS_NAME}\"; export BYOBU_WS_NAME=\"${BYOBU_WS_NAME}\"; export PATH=\"${BYOBU_CONFIG_DIR}/bin:\${PATH}\"; cd ${workdir}; eval \"\$(micromamba shell hook --shell bash)\"; micromamba activate ${MAMBAENVNAME}"
    for extra_cmd in "$@"; do
        setup_cmd+="; $extra_cmd"
    done
    setup_cmd+="; ulimit -n ${N_FILES}"
    execute_command "$setup_cmd"
}

setup_isaac_env_pane() {
    execute_command "export IBRIDO_BYOBU_WS_NAME=\"${IBRIDO_BYOBU_WS_NAME}\"; export BYOBU_WS_NAME=\"${BYOBU_WS_NAME}\"; export PATH=\"${BYOBU_CONFIG_DIR}/bin:\${PATH}\"; cd ${WORKING_DIR}; eval \"\$(micromamba shell hook --shell bash)\"; micromamba activate ${MAMBAENVNAME_ISAAC}; source /isaac-sim/setup_conda_env.sh; source ${WS_ROOT}/setup.bash; export EXP_PATH=/isaac-sim/apps; ulimit -n ${N_FILES}"
}

build_world_cmd() {
    ibrido_build_world_cmd "$@"
    world_cmd="$IBRIDO_WORLD_CMD"
}

build_cluster_cmd() {
    ibrido_build_cluster_cmd "$1" 1 1
    cluster_cmd="$IBRIDO_CLUSTER_CMD"
}

build_training_cmd() {
    ibrido_build_training_cmd
    training_env_cmd="$IBRIDO_TRAINING_CMD"
}

prepare_primary_run_metadata() {
    local run_id="${unique_id:-byobu_$(date '+%Y_%m_%d__%H_%M_%S')}"
    local metadata_world_cmd
    local metadata_cluster_cmd
    local metadata_training_cmd

    export IBRIDO_RUN_META_DIR="${IBRIDO_RUN_META_DIR:-${HOME}/ibrido_logs/ibrido_run_${run_id}/metadata}"

    local metadata_world_iface="${WORLD_INTERFACE:-}"
    local metadata_world_headless=1
    local metadata_world_custom_jnt_imp=1
    local metadata_world_use_gpu="${USE_GPU_SIM:-1}"
    local metadata_world_jnt_imp_config_path="${JNT_IMP_CONFIG_PATH}"

    if [ -z "$metadata_world_iface" ]; then
        echo "launch_byobu_ws.sh: WORLD_INTERFACE is required by the selected cfg"
        exit 1
    fi

    case "$metadata_world_iface" in
        *xmj_world_interface*)
            metadata_world_headless="${XMJ_HEADLESS:-0}"
            metadata_world_custom_jnt_imp=0
            metadata_world_use_gpu=0
            prepare_resolved_xbot_runtime_config
            if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
                metadata_world_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
            fi
            ;;
        *rt_deploy_world_interface*)
            metadata_world_headless="${RT_HEADLESS:-0}"
            metadata_world_custom_jnt_imp=0
            metadata_world_use_gpu=0
            prepare_resolved_xbot_runtime_config
            if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
                metadata_world_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
            fi
            ;;
        *isaac5x_world_interface*)
            metadata_world_headless="${WORLD_HEADLESS:-1}"
            metadata_world_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
            metadata_world_use_gpu="${USE_GPU_SIM:-1}"
            ;;
        *)
            echo "launch_byobu_ws.sh: unsupported WORLD_INTERFACE: $metadata_world_iface"
            exit 1
            ;;
    esac

    ibrido_build_world_cmd "$metadata_world_iface" "$N_ENVS" "$metadata_world_headless" "$metadata_world_custom_jnt_imp" "$metadata_world_use_gpu" "$metadata_world_jnt_imp_config_path"
    metadata_world_cmd="python launch_world_interface.py $IBRIDO_WORLD_CMD"
    ibrido_build_cluster_cmd "$N_ENVS" 1 1
    metadata_cluster_cmd="python launch_control_cluster.py $IBRIDO_CLUSTER_CMD"
    ibrido_build_training_cmd
    metadata_training_cmd="python launch_train_env.py $IBRIDO_TRAINING_CMD --comment \"$COMMENT\""

    ibrido_prepare_run_metadata "$config_file" "$metadata_world_cmd" "$metadata_cluster_cmd" "$metadata_training_cmd" "$run_id"
    record_byobu_aux_metadata

    if enabled "$dry_run"; then
        ibrido_print_dry_run "$metadata_world_cmd" "$metadata_cluster_cmd" "$metadata_training_cmd"
        exit 0
    fi
}

create_execution_layout() {
    split_h
    split_h
    byobu select-layout even-vertical
}

selected_world_interface_matches() {
    [[ "${WORLD_INTERFACE:-}" == *"$1"* ]]
}

prepare_disabled_backend_tab() {
    local backend_label="$1"
    local hint="$2"
    local pane

    create_execution_layout
    for pane in 0 1 2; do
        go_to_pane "$pane"
        setup_main_env_pane "${WORKING_DIR}"
        execute_command "reset && printf '%s\n' \"${backend_label} tab is disabled for this cfg.\" \"cfg: ${config_file}\" \"WORLD_INTERFACE: ${WORLD_INTERFACE:-unset}\" \"${hint}\""
    done
}

prepare_cluster_pane() {
    local cluster_size="$1"

    setup_main_env_pane "${WORKING_DIR}" "source ${WS_ROOT}/setup.bash"
    build_cluster_cmd "$cluster_size"
    prepare_command "reset && python launch_control_cluster.py $cluster_cmd"
}

prepare_gui_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    build_gui_cmd
    prepare_command "reset && $IBRIDO_GUI_CMD"
}

prepare_rhc_cmd_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    build_rhc_cmd
    prepare_command "reset && $IBRIDO_RHC_CMD"
}

prepare_agent_cmd_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    build_agent_cmd
    prepare_command "reset && $IBRIDO_AGENT_CMD"
}

prepare_joy_zmq_pub_pane() {
    setup_main_env_pane "${WORKING_DIR_JOY}" "source ${WS_ROOT}/setup.bash"
    build_joy_zmq_pub_cmd
    prepare_command "reset && $IBRIDO_JOY_ZMQ_PUB_CMD"
}

build_eval_training_cmd() {
    local eval_n_envs="${1:-1}"
    local eval_tot_steps="${2:-${EVAL_TOT_STEPS:-1000}}"
    local saved_eval="${EVAL:-}"
    local saved_resume="${RESUME:-}"
    local saved_n_envs="${N_ENVS:-}"
    local saved_tot_steps="${TOT_STEPS:-}"
    local saved_override_env="${OVERRIDE_ENV:-}"
    local saved_dump_env_checkpoints="${DUMP_ENV_CHECKPOINTS:-}"
    local saved_cluster_db="${CLUSTER_DB:-}"
    local saved_debug="${DEBUG:-}"
    local saved_rmdebug="${RMDEBUG:-}"
    local saved_env_idx_bag="${ENV_IDX_BAG:-}"
    local saved_env_idx_bag_demo="${ENV_IDX_BAG_DEMO:-}"
    local saved_env_idx_bag_expl="${ENV_IDX_BAG_EXPL:-}"

    EVAL=1
    RESUME=0
    N_ENVS="$eval_n_envs"
    TOT_STEPS="$eval_tot_steps"
    OVERRIDE_ENV=1
    DUMP_ENV_CHECKPOINTS=0
    CLUSTER_DB=0
    DEBUG=0
    RMDEBUG=0
    ENV_IDX_BAG=-1
    ENV_IDX_BAG_DEMO=-1
    ENV_IDX_BAG_EXPL=-1

    ibrido_build_training_cmd
    IBRIDO_EVAL_TRAINING_CMD="$IBRIDO_TRAINING_CMD"

    EVAL="$saved_eval"
    RESUME="$saved_resume"
    N_ENVS="$saved_n_envs"
    TOT_STEPS="$saved_tot_steps"
    OVERRIDE_ENV="$saved_override_env"
    DUMP_ENV_CHECKPOINTS="$saved_dump_env_checkpoints"
    CLUSTER_DB="$saved_cluster_db"
    DEBUG="$saved_debug"
    RMDEBUG="$saved_rmdebug"
    ENV_IDX_BAG="$saved_env_idx_bag"
    ENV_IDX_BAG_DEMO="$saved_env_idx_bag_demo"
    ENV_IDX_BAG_EXPL="$saved_env_idx_bag_expl"
}

prepare_training_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    build_training_cmd
    prepare_command "reset && python launch_train_env.py $training_env_cmd --comment \"$COMMENT\""
}

prepare_eval_training_pane() {
    local eval_n_envs="${1:-1}"
    local eval_tot_steps="${2:-${EVAL_TOT_STEPS:-1000}}"

    setup_main_env_pane "${WORKING_DIR}"
    build_eval_training_cmd "$eval_n_envs" "$eval_tot_steps"
    prepare_command "reset && python launch_train_env.py $IBRIDO_EVAL_TRAINING_CMD --comment \"$COMMENT\""
}

prepare_ros_bridge_pane() {
    setup_main_env_pane "${WORKING_DIR}" "source /opt/ros/jazzy/setup.bash" "source ${WS_ROOT}/setup.bash"
    build_ros_bridge_cmd
    prepare_command "reset && $IBRIDO_ROS_BRIDGE_CMD"
}

prepare_bag_dump_pane() {
    setup_main_env_pane "${WORKING_DIR}" "source /opt/ros/jazzy/setup.bash" "source ${WS_ROOT}/setup.bash"
    build_bag_dump_cmd
    prepare_command "reset && $IBRIDO_BAG_DUMP_CMD"
}

build_gui_cmd() {
    IBRIDO_GUI_CMD="python utilities/launch_GUI.py --ns $SHM_NS"
}

build_rhc_cmd() {
    IBRIDO_RHC_CMD="python utilities/launch_rhc_keybrd_cmds.py --ns $SHM_NS --env_idx 0 --from_stdin --add_remote_exit --joy"
}

build_agent_cmd() {
    IBRIDO_AGENT_CMD="python utilities/launch_agent_keybrd_cmds.py --ns $SHM_NS --env_idx 0 --agent_refs_world --from_stdin --add_remote_exit --joy"
}

build_joy_zmq_pub_cmd() {
    IBRIDO_JOY_ZMQ_PUB_CMD="python joy_zmq_pub.py"
}

build_ros_bridge_cmd() {
    local heightmap_arg=""

    if enabled "$PUB_HEIGHTMAP"; then
        heightmap_arg=" --show_heightmap"
    fi
    IBRIDO_ROS_BRIDGE_CMD="python utilities/launch_rhc2rviz_bridge.py --ros2 --rhc_refs_in_h_frame --ns $SHM_NS --with_agent_refs --no_rhc_internal${heightmap_arg}"
}

build_bag_dump_cmd() {
    local heightmap_arg=""

    if enabled "$PUB_HEIGHTMAP"; then
        heightmap_arg=" --show_heightmap"
    fi
    IBRIDO_BAG_DUMP_CMD="python utilities/launch_periodic_bag_dump.py --ros2 --is_training --use_shared_drop_dir --pub_stime --ns $SHM_NS --rhc_refs_in_h_frame --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG --srdf_path $SRDF_PATH_ROSBAG --with_agent_refs --no_rhc_internal${heightmap_arg}"
}

build_zmq_bridge_cmd() {
    local zmq_cmd="--ns $SHM_NS --dt ${ZMQ_BRIDGE_DT:-0.05}"
    local cores_arg="${ZMQ_BRIDGE_CORES:--1}"

    if enabled "${ZMQ_BRIDGE_FULL_DATA:-1}"; then
        zmq_cmd="$zmq_cmd --add_training_data"
    fi
    if enabled "${ZMQ_BRIDGE_RHC_INTERNAL:-1}"; then
        zmq_cmd="$zmq_cmd --add_rhc_internal"
    fi
    if [ -n "${ZMQ_BRIDGE_ENV_IDX:-}" ]; then
        zmq_cmd="$zmq_cmd --env_idx ${ZMQ_BRIDGE_ENV_IDX}"
    fi
    if [ -n "${ZMQ_BRIDGE_BIND_IP:-}" ]; then
        zmq_cmd="$zmq_cmd --bind_ip ${ZMQ_BRIDGE_BIND_IP}"
    fi
    if [ -n "${ZMQ_BRIDGE_SOURCE_IP:-}" ]; then
        zmq_cmd="$zmq_cmd --source_ip ${ZMQ_BRIDGE_SOURCE_IP}"
    fi
    if [ -n "${ZMQ_BRIDGE_PORT_BASE:-}" ]; then
        zmq_cmd="$zmq_cmd --port_base ${ZMQ_BRIDGE_PORT_BASE}"
    fi
    if [ -n "${ZMQ_BRIDGE_PORT_SPAN:-}" ]; then
        zmq_cmd="$zmq_cmd --port_span ${ZMQ_BRIDGE_PORT_SPAN}"
    fi
    if [ -n "${ZMQ_BRIDGE_TIMEOUT_MS:-}" ]; then
        zmq_cmd="$zmq_cmd --timeout_ms ${ZMQ_BRIDGE_TIMEOUT_MS}"
    fi
    if enabled "${ZMQ_BRIDGE_NO_CONFLATE:-0}"; then
        zmq_cmd="$zmq_cmd --no_conflate"
    fi
    if enabled "${ZMQ_BRIDGE_VERBOSE:-0}"; then
        zmq_cmd="$zmq_cmd --verbose"
    fi
    if [ "$cores_arg" != "-1" ]; then
        zmq_cmd="$zmq_cmd --cores ${cores_arg}"
    fi

    IBRIDO_ZMQ_BRIDGE_CMD="python utilities/launch_rhc2zmq_bridge.py $zmq_cmd"
}

record_byobu_launch_command() {
    local name="$1"
    local workdir="$2"
    local cmd="$3"

    ibrido_record_launch_command "$name" "cd ${workdir} && ${cmd}"
}

build_xmj_world_cmd_for_metadata() {
    local xmj_n_envs="${XMJ_N_ENVS:-1}"
    local xmj_headless="${XMJ_HEADLESS:-0}"
    local xmj_jnt_imp_config_path="$JNT_IMP_CONFIG_PATH"
    local xmj_custom_args_names="$CUSTOM_ARGS_NAMES"
    local xmj_custom_args_dtype="$CUSTOM_ARGS_DTYPE"
    local xmj_custom_args_vals="$CUSTOM_ARGS_VALS"

    prepare_resolved_xbot_runtime_config
    resolve_xmj_files_dir

    if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
        xmj_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
    fi

    if [ -n "${RESOLVED_XMJ_FILES_DIR:-}" ] && [[ " ${xmj_custom_args_names} " != *" xmj_files_dir "* ]]; then
        xmj_custom_args_names+=" xmj_files_dir"
        xmj_custom_args_dtype+=" str"
        xmj_custom_args_vals+=" ${WS_ROOT}/src/${RESOLVED_XMJ_FILES_DIR}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" xmj_timeout "* ]]; then
        xmj_custom_args_names+=" xmj_timeout"
        xmj_custom_args_dtype+=" int"
        xmj_custom_args_vals+=" ${XMJ_TIMEOUT_MS:-30000}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" xbot2_sense_timeout_s "* ]]; then
        xmj_custom_args_names+=" xbot2_sense_timeout_s"
        xmj_custom_args_dtype+=" float"
        xmj_custom_args_vals+=" ${XMJ_SENSE_TIMEOUT_S:-2.0}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" add_remote_exit_flag "* ]]; then
        xmj_custom_args_names+=" add_remote_exit_flag"
        xmj_custom_args_dtype+=" bool"
        xmj_custom_args_vals+=" true"
    fi

    build_world_cmd "aug_mpc_envs.world_interfaces.xmj_world_interface" "$xmj_n_envs" "$xmj_headless" 0 0 \
        "$xmj_jnt_imp_config_path" "$xmj_custom_args_names" "$xmj_custom_args_dtype" "$xmj_custom_args_vals"
    IBRIDO_XMJ_WORLD_CMD="python launch_world_interface.py $world_cmd"
}

record_byobu_aux_metadata() {
    local rt_n_envs="${RT_N_ENVS:-1}"
    local xmj_n_envs="${XMJ_N_ENVS:-1}"
    local rt_world_cmd
    local rt_sim_cmd
    local rt_cluster_cmd
    local rt_training_cmd
    local rt_jnt_imp_config_path="$JNT_IMP_CONFIG_PATH"
    local rt_custom_args_names="$CUSTOM_ARGS_NAMES"
    local rt_custom_args_dtype="$CUSTOM_ARGS_DTYPE"
    local rt_custom_args_vals="$CUSTOM_ARGS_VALS"
    local xmj_cluster_cmd
    local xmj_training_cmd
    local mpcviz_heightmap_arg=""

    if enabled "$PUB_HEIGHTMAP"; then
        mpcviz_heightmap_arg=" --show_heightmap"
    fi

    prepare_resolved_xbot_runtime_config
    if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
        rt_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
    fi
    if [[ " ${rt_custom_args_names} " != *" is_sim "* ]]; then
        rt_custom_args_names+=" is_sim"
        rt_custom_args_dtype+=" bool"
        rt_custom_args_vals+=" true"
    fi
    if [[ " ${rt_custom_args_names} " != *" time_source "* ]]; then
        rt_custom_args_names+=" time_source"
        rt_custom_args_dtype+=" str"
        rt_custom_args_vals+=" sim"
    fi
    if [[ " ${rt_custom_args_names} " != *" add_remote_exit_flag "* ]]; then
        rt_custom_args_names+=" add_remote_exit_flag"
        rt_custom_args_dtype+=" bool"
        rt_custom_args_vals+=" true"
    fi
    build_world_cmd "aug_mpc_envs.world_interfaces.rt_deploy_world_interface" "$rt_n_envs" 0 0 0 \
        "$rt_jnt_imp_config_path" "$rt_custom_args_names" "$rt_custom_args_dtype" "$rt_custom_args_vals"
    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        rt_world_cmd="echo 'XBOT_CONFIG is not set in the selected cfg'"
    else
        rt_world_cmd="set_xbot2_config \"${RESOLVED_XBOT_CONFIG_PATH}\" && python launch_world_interface.py $world_cmd"
    fi
    record_byobu_launch_command "rt_deploy_world_interface" "$WORKING_DIR" "$rt_world_cmd"

    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        record_byobu_launch_command "xbot2_core_simtime" "$WORKING_DIR" "echo 'XBOT_CONFIG is not set in the selected cfg'"
    else
        record_byobu_launch_command "xbot2_core_simtime" "$WORKING_DIR" "source /opt/ros/jazzy/setup.bash && source /opt/xbot/setup.sh && xbot2-core -S -C ${RESOLVED_XBOT_CONFIG_PATH}"
    fi

    resolve_rt_xmj_launcher
    if [ -z "${RESOLVED_RT_XMJ_CMD:-}" ]; then
        rt_sim_cmd="echo 'No RT XMJ launch command resolved for this cfg. Set RT_XMJ_CMD or RT_XMJ_LAUNCH_CMD.'"
    else
        rt_sim_cmd="$RESOLVED_RT_XMJ_CMD"
    fi
    record_byobu_launch_command "rt_deploy_xmj_sim" "$RESOLVED_RT_XMJ_WORKDIR" "$rt_sim_cmd"

    ibrido_build_cluster_cmd "$rt_n_envs" 1 1
    rt_cluster_cmd="python launch_control_cluster.py $IBRIDO_CLUSTER_CMD"
    record_byobu_launch_command "rt_deploy_control_cluster" "$WORKING_DIR" "$rt_cluster_cmd"

    build_eval_training_cmd "$rt_n_envs"
    rt_training_cmd="python launch_train_env.py $IBRIDO_EVAL_TRAINING_CMD --comment \"$COMMENT\""
    record_byobu_launch_command "rt_deploy_train_env" "$WORKING_DIR" "$rt_training_cmd"

    build_xmj_world_cmd_for_metadata
    record_byobu_launch_command "xmj_world_interface" "$WORKING_DIR" "$IBRIDO_XMJ_WORLD_CMD"

    ibrido_build_cluster_cmd "$xmj_n_envs" 1 1
    xmj_cluster_cmd="python launch_control_cluster.py $IBRIDO_CLUSTER_CMD"
    record_byobu_launch_command "xmj_control_cluster" "$WORKING_DIR" "$xmj_cluster_cmd"

    build_eval_training_cmd "$xmj_n_envs"
    xmj_training_cmd="python launch_train_env.py $IBRIDO_EVAL_TRAINING_CMD --comment \"$COMMENT\""
    record_byobu_launch_command "xmj_train_env" "$WORKING_DIR" "$xmj_training_cmd"

    build_gui_cmd
    record_byobu_launch_command "debug_gui" "$WORKING_DIR" "$IBRIDO_GUI_CMD"
    build_ros_bridge_cmd
    record_byobu_launch_command "debug_ros_bridge" "$WORKING_DIR" "$IBRIDO_ROS_BRIDGE_CMD"
    build_bag_dump_cmd
    record_byobu_launch_command "debug_bag_dump" "$WORKING_DIR" "$IBRIDO_BAG_DUMP_CMD"
    build_zmq_bridge_cmd
    record_byobu_launch_command "debug_zmq_bridge" "$WORKING_DIR" "$IBRIDO_ZMQ_BRIDGE_CMD"

    mpcviz_custom_xacro_args_for centauro
    record_byobu_launch_command "mpcviz_centauro" "$WORKING_DIR_CENTAURO" \
        "python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}${mpcviz_heightmap_arg}"
    mpcviz_custom_xacro_args_for kyon02 kyon_simple
    record_byobu_launch_command "mpcviz_kyon02" "$WORKING_DIR_QUAD" \
        "python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --kyon_real --wheels${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}${mpcviz_heightmap_arg}"
    mpcviz_custom_xacro_args_for b2w
    record_byobu_launch_command "mpcviz_b2w" "$WORKING_DIR_QUAD" \
        "python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --b2w${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}${mpcviz_heightmap_arg}"
    mpcviz_custom_xacro_args_for talos
    record_byobu_launch_command "mpcviz_talos" "$WORKING_DIR_TALOS" \
        "python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}${mpcviz_heightmap_arg}"

    build_joy_zmq_pub_cmd
    record_byobu_launch_command "teleop_joy_zmq_pub" "$WORKING_DIR_JOY" "$IBRIDO_JOY_ZMQ_PUB_CMD"
    build_rhc_cmd
    record_byobu_launch_command "teleop_rhc_cmds" "$WORKING_DIR" "$IBRIDO_RHC_CMD"
    build_agent_cmd
    record_byobu_launch_command "teleop_agent_cmds" "$WORKING_DIR" "$IBRIDO_AGENT_CMD"

    ibrido_record_launch_command "resources_htop" "htop"
    ibrido_record_launch_command "resources_nvtop" "nvtop"
    ibrido_write_launch_index "${IBRIDO_RUN_META_DIR}"
}

prepare_zmq_bridge_pane() {
    build_zmq_bridge_cmd
    setup_main_env_pane "${WORKING_DIR}" "source ${WS_ROOT}/setup.bash"
    prepare_command "reset && $IBRIDO_ZMQ_BRIDGE_CMD"
}

add_isaac5x_tab() {
    if ! selected_world_interface_matches "isaac5x_world_interface"; then
        prepare_disabled_backend_tab "isaac5x" "Use an isaac5x run profile for this tab."
        return
    fi

    create_execution_layout

    go_to_pane 0
    setup_isaac_env_pane
    build_world_cmd "aug_mpc_envs.world_interfaces.isaac5x_world_interface" "$N_ENVS" 1 1 "$USE_GPU_SIM"
    prepare_command "reset && python launch_world_interface.py $world_cmd"

    go_to_pane 1
    prepare_cluster_pane "$N_ENVS"

    go_to_pane 2
    prepare_training_pane
}

resolve_xbot_config() {
    local cfg_key

    RESOLVED_XBOT_CONFIG="${XBOT_CONFIG:-}"
    if [ -n "$RESOLVED_XBOT_CONFIG" ]; then
        return
    fi

    cfg_key="${ROBOT_FAMILY:-} ${ROBOT_VARIANT:-} ${SHM_NS:-} ${RNAME:-} ${URDF_PATH:-} ${CLUSTER_CL_FNAME:-} ${CUSTOM_ARGS_VALS:-}"
    cfg_key="${cfg_key,,}"

    if [[ "$cfg_key" == *centauro* ]]; then
        RESOLVED_XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/xbot2_basic.yaml"
    elif [[ "$cfg_key" == *b2w* ]]; then
        RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w/xbot2_basic.yaml"
    elif [[ "$cfg_key" == *kyon* ]]; then
        if [[ "$cfg_key" == *iit-kyon-ros-pkg* || "$cfg_key" == *kyon_simple* ]]; then
            if [[ "$cfg_key" == *wheels* && "$cfg_key" != *no_wheels* ]]; then
                RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic_wheels.yaml"
            else
                RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic.yaml"
            fi
        elif [[ "$cfg_key" == *wheels_no_yaw* || "$cfg_key" == *no_yaw* ]]; then
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic_wheels_no_yaw.yaml"
        elif [[ "$cfg_key" == *wheels* && "$cfg_key" != *no_wheels* ]]; then
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic_wheels.yaml"
        else
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic.yaml"
        fi
    elif [[ "$cfg_key" == *talos* ]]; then
        RESOLVED_XBOT_CONFIG="TalosHybridMPC/taloshybridmpc/config/xmj_env_files/xbot2_basic.yaml"
    fi
}

prepare_resolved_xbot_runtime_config() {
    local saved_ws_src="${IBRIDO_WS_SRC:-}"

    resolve_xbot_config
    RESOLVED_XBOT_CONFIG_PATH=""
    RESOLVED_XBOT_TEMPLATE_CONFIG_PATH=""
    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        return
    fi

    export IBRIDO_WS_SRC="${WS_ROOT}/src"
    ibrido_prepare_xbot_runtime_config "$RESOLVED_XBOT_CONFIG" "$JNT_IMP_CONFIG_PATH" || exit 1
    RESOLVED_XBOT_CONFIG_PATH="$IBRIDO_XBOT_RUNTIME_CONFIG_PATH"
    RESOLVED_XBOT_TEMPLATE_CONFIG_PATH="$IBRIDO_XBOT_TEMPLATE_CONFIG_PATH"

    if [ -n "$saved_ws_src" ]; then
        export IBRIDO_WS_SRC="$saved_ws_src"
    else
        unset IBRIDO_WS_SRC
    fi
}

prepare_xbot2_gui_config() {
    local gui_cfg_dir
    local launcher_cfg
    local server_cfg
    local runner
    local gui_home

    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        RESOLVED_XBOT2_GUI_SERVER_CONFIG=""
        RESOLVED_XBOT2_GUI_LAUNCHER_CONFIG=""
        RESOLVED_XBOT2_GUI_RUNNER=""
        return
    fi

    gui_home="${CONTAINER_HOME:-${HOME}}"
    gui_cfg_dir="${gui_home}/.xbot/ibrido/xbot2_gui"
    launcher_cfg="${gui_cfg_dir}/launcher_config.yaml"
    server_cfg="${gui_cfg_dir}/server_config.yaml"
    runner="${gui_cfg_dir}/run_xbot2_gui.sh"

    mkdir -p "${gui_cfg_dir}"

    cat > "${launcher_cfg}" <<EOF
context:
  session: ibrido_xbot2
EOF

cat > "${server_cfg}" <<EOF
extensions:
  - module: xbot2_gui_server.joint_states
    class: JointStateHandler
  - module: xbot2_gui_server.joint_device
    class: JointDeviceHandler
  - module: xbot2_gui_server.plugin
    class: PluginHandler
  - module: xbot2_gui_server.launcher
    class: Launcher
  - module: xbot2_gui_server.visual
    class: VisualHandler
  - module: xbot2_gui_server.server_stats
    class: ServerStatisticsHandler

launcher:
  type: launcher
  launcher_config: launcher_config.yaml
  cache_path: ${gui_cfg_dir}/launcher_cache.yaml

joint_states:
  rate: 30.0

visual:
  rate: 10.0
EOF

    cat > "${runner}" <<EOF
#!/usr/bin/env bash
set -e

mkdir -p "${gui_cfg_dir}"
cat > "${launcher_cfg}" <<'LAUNCHER_EOF'
context:
  session: ibrido_xbot2
LAUNCHER_EOF

cat > "${server_cfg}" <<'SERVER_EOF'
extensions:
  - module: xbot2_gui_server.joint_states
    class: JointStateHandler
  - module: xbot2_gui_server.joint_device
    class: JointDeviceHandler
  - module: xbot2_gui_server.plugin
    class: PluginHandler
  - module: xbot2_gui_server.launcher
    class: Launcher
  - module: xbot2_gui_server.visual
    class: VisualHandler
  - module: xbot2_gui_server.server_stats
    class: ServerStatisticsHandler

launcher:
  type: launcher
  launcher_config: launcher_config.yaml
  cache_path: ${gui_cfg_dir}/launcher_cache.yaml

joint_states:
  rate: 30.0

visual:
  rate: 10.0
SERVER_EOF

export XBOT2_GUI_SERVER_CONFIG="${server_cfg}"
exec xbot2-gui
EOF
    chmod +x "${runner}"

    RESOLVED_XBOT2_GUI_SERVER_CONFIG="${server_cfg}"
    RESOLVED_XBOT2_GUI_LAUNCHER_CONFIG="${launcher_cfg}"
    RESOLVED_XBOT2_GUI_RUNNER="${runner}"
}

resolve_xmj_files_dir() {
    RESOLVED_XMJ_FILES_DIR="${XMJ_FILES_DIR:-}"
    if [ -n "$RESOLVED_XMJ_FILES_DIR" ]; then
        return
    fi

    resolve_xbot_config
    if [ -n "$RESOLVED_XBOT_CONFIG" ]; then
        RESOLVED_XMJ_FILES_DIR="${RESOLVED_XBOT_CONFIG%/*}"
    fi
}

resolve_rt_xmj_launcher() {
    local cfg_key
    local rt_factor="${RT_XMJ_RT_FACTOR:-1.0}"
    local ros_version="${RT_XMJ_ROS_VERSION:-ros2}"
    local rostime_args=""

    if enabled "${RT_XMJ_PUB_ROSTIME:-0}"; then
        rostime_args=" --pub-rostime --ros-version ${ros_version}"
    fi

    RESOLVED_RT_XMJ_WORKDIR="${RT_XMJ_WORKDIR:-}"
    RESOLVED_RT_XMJ_CMD="${RT_XMJ_CMD:-${RT_XMJ_LAUNCH_CMD:-}}"
    if [ -n "$RESOLVED_RT_XMJ_CMD" ]; then
        RESOLVED_RT_XMJ_WORKDIR="${RESOLVED_RT_XMJ_WORKDIR:-$WORKING_DIR}"
        return
    fi

    cfg_key="${ROBOT_FAMILY:-} ${ROBOT_VARIANT:-} ${SHM_NS:-} ${RNAME:-} ${URDF_PATH:-} ${CLUSTER_CL_FNAME:-} ${CUSTOM_ARGS_VALS:-}"
    cfg_key="${cfg_key,,}"

    if [[ "$cfg_key" == *centauro* ]]; then
        RESOLVED_RT_XMJ_WORKDIR="$WORKING_DIR_CENTAURO"
        RESOLVED_RT_XMJ_CMD="./launch_xmj_centauro.sh --rt_factor ${rt_factor}${rostime_args}"
    elif [[ "$cfg_key" == *b2w* || "$cfg_key" == *unitree* ]]; then
        local b2w_xmj_dir="${WS_ROOT}/src/KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w"
        local b2w_xbot_config="${RESOLVED_XBOT_CONFIG_PATH:-${WS_ROOT}/src/KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w/xbot2_basic.yaml}"
        local b2w_urdf_path="${IBRIDO_XBOT_RUNTIME_URDF_PATH:-${b2w_xmj_dir}/unitree_b2w.urdf}"
        local b2w_root_spawn_height="${RT_XMJ_ROOT_SPAWN_HEIGHT:-$(custom_arg_value root_spawn_height || true)}"
        local b2w_root_spawn_args=""
        if [ -n "${b2w_root_spawn_height:-}" ]; then
            b2w_root_spawn_args=" --root_spawn_height ${b2w_root_spawn_height}"
        fi
        RESOLVED_RT_XMJ_WORKDIR="${WS_ROOT}/src/xbot2_mujoco/tests/PyXBotMjSim"
        RESOLVED_RT_XMJ_CMD="python launch_simulator.py --files_dir ${b2w_xmj_dir} --urdf_path ${b2w_urdf_path} --simopt_path ${b2w_xmj_dir}/sim_opt.xml --world_path ${b2w_xmj_dir}/world.xml --sites_path ${b2w_xmj_dir}/sites.xml --xbot_config_path ${b2w_xbot_config} --blink_name base --rt_factor ${rt_factor}${b2w_root_spawn_args}${rostime_args}"
    elif [[ "$cfg_key" == *kyon* ]]; then
        RESOLVED_RT_XMJ_WORKDIR="$WORKING_DIR_QUAD"
        if [[ "$cfg_key" == *no_wheels* ]]; then
            RESOLVED_RT_XMJ_CMD="./launch_xmj_kyon_real_no_wheels.sh --rt_factor ${rt_factor}${rostime_args}"
        else
            RESOLVED_RT_XMJ_CMD="./launch_xmj_kyon_real_wheels.sh --rt_factor ${rt_factor}${rostime_args}"
        fi
    elif [[ "$cfg_key" == *talos* ]]; then
        RESOLVED_RT_XMJ_WORKDIR="$WORKING_DIR_TALOS"
        RESOLVED_RT_XMJ_CMD="./launch_xmj_talos.sh --rt_factor ${rt_factor}${rostime_args}"
    else
        RESOLVED_RT_XMJ_WORKDIR="$WORKING_DIR"
        RESOLVED_RT_XMJ_CMD=""
    fi
}

prepare_rt_xmj_sim_pane() {
    resolve_rt_xmj_launcher
    setup_main_env_pane "${RESOLVED_RT_XMJ_WORKDIR}" \
        "source /opt/xbot/setup.sh" \
        "source ${WS_ROOT}/setup.bash"

    if [ -z "${RESOLVED_RT_XMJ_CMD:-}" ]; then
        prepare_command "echo 'No RT XMJ launch command resolved for this cfg. Set RT_XMJ_CMD or RT_XMJ_LAUNCH_CMD.'"
    else
        prepare_command "${RESOLVED_RT_XMJ_CMD}"
    fi
}

add_rt_deployment_tab() {
    local rt_n_envs="${RT_N_ENVS:-1}"
    local rt_jnt_imp_config_path="$JNT_IMP_CONFIG_PATH"
    local rt_custom_args_names="$CUSTOM_ARGS_NAMES"
    local rt_custom_args_dtype="$CUSTOM_ARGS_DTYPE"
    local rt_custom_args_vals="$CUSTOM_ARGS_VALS"
    local rt_world_pane
    local rt_sim_pane
    local rt_cluster_pane
    local rt_training_pane

    if ! selected_world_interface_matches "rt_deploy_world_interface"; then
        prepare_disabled_backend_tab "rt_deployment" "Use an rt_xbot_zmq run profile for this tab."
        return
    fi

    create_execution_layout

    go_to_pane 1
    rt_cluster_pane="$(current_pane_id)"

    go_to_pane 2
    rt_training_pane="$(current_pane_id)"

    go_to_pane 0
    rt_world_pane="$(current_pane_id)"
    split_v
    rt_sim_pane="$(current_pane_id)"

    go_to_pane "$rt_world_pane"
    prepare_resolved_xbot_runtime_config
    if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
        rt_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
    fi
    if [[ " ${rt_custom_args_names} " != *" is_sim "* ]]; then
        rt_custom_args_names+=" is_sim"
        rt_custom_args_dtype+=" bool"
        rt_custom_args_vals+=" true"
    fi
    if [[ " ${rt_custom_args_names} " != *" time_source "* ]]; then
        rt_custom_args_names+=" time_source"
        rt_custom_args_dtype+=" str"
        rt_custom_args_vals+=" sim"
    fi
    if [[ " ${rt_custom_args_names} " != *" add_remote_exit_flag "* ]]; then
        rt_custom_args_names+=" add_remote_exit_flag"
        rt_custom_args_dtype+=" bool"
        rt_custom_args_vals+=" true"
    fi
    setup_main_env_pane "${WORKING_DIR}" \
        "source ${WS_ROOT}/setup.bash"
    build_world_cmd "aug_mpc_envs.world_interfaces.rt_deploy_world_interface" "$rt_n_envs" 0 0 0 \
        "$rt_jnt_imp_config_path" "$rt_custom_args_names" "$rt_custom_args_dtype" "$rt_custom_args_vals"
    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        prepare_command "reset && echo 'XBOT_CONFIG is not set in the selected cfg'"
    else
        prepare_command "reset && set_xbot2_config \"${RESOLVED_XBOT_CONFIG_PATH}\" && python launch_world_interface.py $world_cmd"
    fi

    go_to_pane "$rt_sim_pane"
    prepare_rt_xmj_sim_pane

    go_to_pane "$rt_cluster_pane"
    prepare_cluster_pane "$rt_n_envs"

    go_to_pane "$rt_training_pane"
    prepare_eval_training_pane "$rt_n_envs"
}

add_xbot2_tab() {
    local xbot_cfg_path
    local xbot_cfg_dir

    prepare_resolved_xbot_runtime_config
    prepare_xbot2_gui_config

    setup_main_env_pane "${WORKING_DIR}" \
        "source /opt/ros/jazzy/setup.bash" \
        "source /opt/xbot/setup.sh" \
        "source ${WS_ROOT}/setup.bash"

    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        prepare_command "reset && echo 'XBOT_CONFIG is not set in the selected cfg'"
    else
        xbot_cfg_path="${RESOLVED_XBOT_CONFIG_PATH}"
        xbot_cfg_dir="$(dirname "${xbot_cfg_path}")"
        prepare_command "reset && cd ${xbot_cfg_dir} && xbot2-core -S -C ${xbot_cfg_path}"
    fi

    split_h
    setup_main_env_pane "${WORKING_DIR}" \
        "source /opt/ros/jazzy/setup.bash" \
        "source /opt/xbot/setup.sh" \
        "source ${WS_ROOT}/setup.bash"
    if [ -z "${RESOLVED_XBOT2_GUI_RUNNER:-}" ]; then
        prepare_command "reset && xbot2-gui"
    else
        prepare_command "reset && ${RESOLVED_XBOT2_GUI_RUNNER}"
    fi
}

add_xmj_tab() {
    local xmj_n_envs="${XMJ_N_ENVS:-1}"
    local xmj_headless="${XMJ_HEADLESS:-0}"
    local xmj_jnt_imp_config_path="$JNT_IMP_CONFIG_PATH"
    local xmj_custom_args_names="$CUSTOM_ARGS_NAMES"
    local xmj_custom_args_dtype="$CUSTOM_ARGS_DTYPE"
    local xmj_custom_args_vals="$CUSTOM_ARGS_VALS"

    if ! selected_world_interface_matches "xmj_world_interface"; then
        prepare_disabled_backend_tab "xmj" "Use an xmj run profile for this tab."
        return
    fi

    prepare_resolved_xbot_runtime_config
    resolve_xmj_files_dir

    if [ -n "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
        xmj_jnt_imp_config_path="${RESOLVED_XBOT_CONFIG_PATH}"
    fi

    if [ -n "${RESOLVED_XMJ_FILES_DIR:-}" ] && [[ " ${xmj_custom_args_names} " != *" xmj_files_dir "* ]]; then
        xmj_custom_args_names+=" xmj_files_dir"
        xmj_custom_args_dtype+=" str"
        xmj_custom_args_vals+=" ${WS_ROOT}/src/${RESOLVED_XMJ_FILES_DIR}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" xmj_timeout "* ]]; then
        xmj_custom_args_names+=" xmj_timeout"
        xmj_custom_args_dtype+=" int"
        xmj_custom_args_vals+=" ${XMJ_TIMEOUT_MS:-30000}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" xbot2_sense_timeout_s "* ]]; then
        xmj_custom_args_names+=" xbot2_sense_timeout_s"
        xmj_custom_args_dtype+=" float"
        xmj_custom_args_vals+=" ${XMJ_SENSE_TIMEOUT_S:-2.0}"
    fi
    if [[ " ${xmj_custom_args_names} " != *" add_remote_exit_flag "* ]]; then
        xmj_custom_args_names+=" add_remote_exit_flag"
        xmj_custom_args_dtype+=" bool"
        xmj_custom_args_vals+=" true"
    fi

    create_execution_layout

    go_to_pane 0
    setup_main_env_pane "${WORKING_DIR}" \
        "source ${WS_ROOT}/setup.bash"
    build_world_cmd "aug_mpc_envs.world_interfaces.xmj_world_interface" "$xmj_n_envs" "$xmj_headless" 0 0 \
        "$xmj_jnt_imp_config_path" "$xmj_custom_args_names" "$xmj_custom_args_dtype" "$xmj_custom_args_vals"
    prepare_command "reset && python launch_world_interface.py $world_cmd"

    go_to_pane 1
    prepare_cluster_pane "$xmj_n_envs"

    go_to_pane 2
    prepare_eval_training_pane "$xmj_n_envs"
}

prepare_mpcviz_pane() {
    local workdir="$1"
    local mpcviz_cmd="$2"
    local heightmap_arg=""

    if enabled "$PUB_HEIGHTMAP"; then
        heightmap_arg=" --show_heightmap"
    fi

    setup_main_env_pane "$workdir" "source /opt/ros/jazzy/setup.bash"
    prepare_command "reset && ${mpcviz_cmd}${heightmap_arg}"
}

add_mpcviz_tab() {
    local mpcviz_cmd

    mpcviz_custom_xacro_args_for centauro
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}"
    prepare_mpcviz_pane "${WORKING_DIR_CENTAURO}" "$mpcviz_cmd"

    split_v
    mpcviz_custom_xacro_args_for kyon02 kyon_simple
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --kyon_real --wheels${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}"
    prepare_mpcviz_pane "${WORKING_DIR_QUAD}" "$mpcviz_cmd"

    split_h
    mpcviz_custom_xacro_args_for b2w
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --b2w${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}"
    prepare_mpcviz_pane "${WORKING_DIR_QUAD}" "$mpcviz_cmd"

    go_to_pane 0
    split_h
    mpcviz_custom_xacro_args_for talos
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10${IBRIDO_MPCVIZ_CUSTOM_XACRO_ARGS}"
    prepare_mpcviz_pane "${WORKING_DIR_TALOS}" "$mpcviz_cmd"
}

add_debug_tab() {
    prepare_gui_pane

    split_v
    prepare_ros_bridge_pane

    split_h
    prepare_bag_dump_pane

    go_to_pane 0
    split_h
    prepare_zmq_bridge_pane
}

add_teleop_tab() {
    prepare_joy_zmq_pub_pane

    split_h
    prepare_rhc_cmd_pane

    split_v
    prepare_agent_cmd_pane
}

add_resource_monitoring_tab() {
    install_terminal_helpers
    clear_terminal
    execute_command "htop"

    split_h
    install_terminal_helpers
    execute_command "cd ${WORKING_DIR}"
    clear_terminal
    execute_command "nvtop"
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -cfg|--cfg)
      if [[ "$2" = /* ]]; then
        config_file="$2"
      else
        config_file="${cfg_file_basepath}/$2"
      fi
      shift
      ;;
    --set) cfg_overrides+=("$2"); shift ;;
    --dry-run) dry_run=1 ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -z "$config_file" ]; then
    echo "launch_byobu_ws.sh: --cfg is required"
    usage
fi

if [ -f "$config_file" ]; then
    config_exports="$("${UTILS_DIR}/ibrido_config_loader.py" --shell "$config_file")" || exit 1
    eval "$config_exports"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

for cfg_override in "${cfg_overrides[@]}"; do
  cfg_override_name="${cfg_override%%=*}"
  if [[ "$cfg_override" != *=* || ! "$cfg_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "Invalid --set override: $cfg_override"
    exit 1
  fi
  export "$cfg_override"
done

ibrido_normalize_runtime_config || exit 1
ibrido_validate_runtime_config || exit 1

echo "Will preload script cmds from $config_file"

N_FILES=${ULIM_N:-131072}
prepare_primary_run_metadata

install_byobu_helper_commands

byobu kill-session -t "${BYOBU_WS_NAME}" 2>/dev/null || true
byobu new-session -d -s "${BYOBU_WS_NAME}" -c "${WORKING_DIR}" -n isaac5x

add_isaac5x_tab

new_tab rt_deployment
add_rt_deployment_tab

new_tab xmj
add_xmj_tab

new_tab debug
add_debug_tab

new_tab mpcviz
add_mpcviz_tab

new_tab teleop
add_teleop_tab

new_tab xbot2
add_xbot2_tab

new_tab resources
add_resource_monitoring_tab

attach_to_session
