#!/bin/bash

usage() {
  echo "Usage: $0
    [--cfg CFG] \
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

MAMBAENVNAME="${MAMBA_ENV_NAME}"
MAMBAENVNAME_ISAAC="${MAMBA_ENV_NAME_ISAAC}"

# Default configuration file
config_file="$HOME/ibrido_files/training_cfg.sh"

press_enter() {
    byobu send-keys Enter
    sleep $SLEEP_FOR
}

execute_command() {
    byobu send-keys "$1"
    press_enter
    sleep $SLEEP_FOR
}

prepare_command() {
    clear_terminal
    byobu send-keys "$1"
    sleep $SLEEP_FOR
}

go_to_pane() {
    byobu select-pane -t "$1"
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
    execute_command "reset"
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
    execute_command "export IBRIDO_BYOBU_WS_NAME=\"${IBRIDO_BYOBU_WS_NAME}\""
    execute_command "export BYOBU_WS_NAME=\"${BYOBU_WS_NAME}\""
    execute_command "export PATH=\"${BYOBU_CONFIG_DIR}/bin:\${PATH}\""
}

setup_main_env_pane() {
    local workdir="$1"
    shift

    install_terminal_helpers
    execute_command "cd ${workdir}"
    activate_mamba_env
    for setup_cmd in "$@"; do
        execute_command "$setup_cmd"
    done
    increase_file_limits_locally
}

setup_isaac_env_pane() {
    install_terminal_helpers
    execute_command "cd ${WORKING_DIR}"
    activate_mamba_env_isaac
    execute_command "source /isaac-sim/setup_conda_env.sh"
    execute_command "source ${WS_ROOT}/setup.bash"
    execute_command "export EXP_PATH=/isaac-sim/apps"
    increase_file_limits_locally
}

build_world_cmd() {
    local world_iface_fname="$1"
    local num_envs="$2"
    local headless="$3"
    local use_custom_jnt_imp="$4"
    local use_gpu="$5"
    local jnt_imp_config_path="${6:-$JNT_IMP_CF_PATH}"
    local custom_args_names="${7:-$CUSTOM_ARGS_NAMES}"
    local custom_args_dtype="${8:-$CUSTOM_ARGS_DTYPE}"
    local custom_args_vals="${9:-$CUSTOM_ARGS_VALS}"

    world_cmd="--robot_name $SHM_NS \
--urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
--jnt_imp_config_path $jnt_imp_config_path \
--cluster_dt $CLUSTER_DT \
--physics_dt $PHYSICS_DT \
--n_contacts ${N_CONTACTS:-4} \
--num_envs $num_envs --seed $SEED --timeout_ms $TIMEOUT_MS \
--world_iface_fname $world_iface_fname \
--custom_args_names $custom_args_names \
--custom_args_dtype $custom_args_dtype \
--custom_args_vals $custom_args_vals "

    if enabled "$headless"; then
        world_cmd="--headless $world_cmd"
    fi
    if enabled "$use_custom_jnt_imp"; then
        world_cmd+="--use_custom_jnt_imp "
    fi
    if enabled "$REMOTE_STEPPING"; then
        world_cmd+="--remote_stepping "
    fi
    if enabled "$use_gpu"; then
        world_cmd+="--use_gpu "
    fi
}

build_cluster_cmd() {
    local cluster_size="$1"

    cluster_cmd="--ns $SHM_NS --size $cluster_size --timeout_ms $TIMEOUT_MS \
--urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--cluster_dt $CLUSTER_DT \
--n_nodes $N_NODES "

    if enabled "$CLUSTER_DB"; then
        cluster_cmd+="--enable_debug "
    fi
    if enabled "$IS_CLOSED_LOOP"; then
        cluster_cmd+="--cloop "
    fi
}

build_training_cmd() {
    training_env_cmd="--dump_checkpoints --ns $SHM_NS --drop_dir $HOME/training_data \
--seed $SEED --timeout_ms $TIMEOUT_MS \
--reset_on_init \
--env_fname $TRAIN_ENV_FNAME --env_classname $TRAIN_ENV_CNAME \
--demo_stop_thresh $DEMO_STOP_THRESH  \
--actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH \
--critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH \
--tot_tsteps $TOT_STEPS \
--demo_envs_perc $DEMO_ENVS_PERC \
--expl_envs_perc $EXPL_ENVS_PERC \
--action_repeat $ACTION_REPEAT \
--compression_ratio $COMPRESSION_RATIO \
--discount_factor $DISCOUNT_FACTOR "

    if enabled "$USE_DUMMY"; then
        training_env_cmd+="--dummy "
    elif enabled "$USE_SAC"; then
        training_env_cmd+="--sac "
    fi
    if enabled "$DEBUG"; then
        training_env_cmd+="--db --env_db "
    fi
    if enabled "$RMDEBUG"; then
        training_env_cmd+="--rmdb "
    fi
    if enabled "$DUMP_ENV_CHECKPOINTS" && enabled "$DEBUG"; then
        training_env_cmd+="--full_env_db "
    fi
    if enabled "$USE_RND"; then
        training_env_cmd+="--use_rnd "
    fi
    if enabled "$OBS_NORM"; then
        training_env_cmd+="--obs_norm "
    fi
    if enabled "$OBS_RESCALING"; then
        training_env_cmd+="--obs_rescale "
    fi
    if enabled "$WEIGHT_NORM"; then
        training_env_cmd+="--add_weight_norm "
    fi
    if enabled "$LAYER_NORM"; then
        training_env_cmd+="--add_layer_norm "
    fi
    if enabled "$BATCH_NORM"; then
        training_env_cmd+="--add_batch_norm "
    fi
    if enabled "$CRITIC_ACTION_RESCALE"; then
        training_env_cmd+="--act_rescale_critic "
    fi
    if enabled "$USE_PERIOD_RESETS"; then
        training_env_cmd+="--use_period_resets "
    fi
    if [[ -n "$RNAME" ]]; then
        training_env_cmd+="--run_name ${RNAME}_${TRAIN_ENV_CNAME} "
    fi
    if enabled "$RESUME"; then
        training_env_cmd+="--resume --mpath $MPATH --mname $MNAME "
        if enabled "$OVERRIDE_ENV"; then
            training_env_cmd+="--override_env "
        fi
    fi
    if enabled "$EVAL"; then
        training_env_cmd+="--eval --n_eval_timesteps $TOT_STEPS --mpath $MPATH --mname $MNAME "
        if enabled "$DET_EVAL"; then
            training_env_cmd+="--det_eval "
        fi
        if enabled "$EVAL_ON_CPU"; then
            training_env_cmd+="--use_cpu "
        fi
        if enabled "$OVERRIDE_ENV"; then
            training_env_cmd+="--override_env "
        fi
        if enabled "$OVERRIDE_AGENT_REFS"; then
            training_env_cmd+="--override_agent_refs "
        fi
    fi
}

create_execution_layout() {
    split_h
    split_h
    byobu select-layout even-vertical
}

prepare_cluster_pane() {
    local cluster_size="$1"

    setup_main_env_pane "${WORKING_DIR}" "source ${WS_ROOT}/setup.bash"
    build_cluster_cmd "$cluster_size"
    prepare_command "reset && python launch_control_cluster.py $cluster_cmd"
}

prepare_gui_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    prepare_command "reset && python utilities/launch_GUI.py --ns $SHM_NS"
}

prepare_rhc_cmd_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    prepare_command "reset && python utilities/launch_rhc_keybrd_cmds.py --ns $SHM_NS --env_idx 0 --from_stdin --add_remote_exit --joy"
}

prepare_agent_cmd_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    prepare_command "reset && python utilities/launch_agent_keybrd_cmds.py --ns $SHM_NS --env_idx 0 --agent_refs_world --from_stdin --add_remote_exit --joy"
}

prepare_joy_zmq_pub_pane() {
    setup_main_env_pane "${WORKING_DIR_JOY}" "source ${WS_ROOT}/setup.bash"
    prepare_command "reset && python joy_zmq_pub.py"
}

prepare_training_pane() {
    setup_main_env_pane "${WORKING_DIR}"
    build_training_cmd
    prepare_command "reset && python launch_train_env.py $training_env_cmd --comment \"$COMMENT\""
}

prepare_ros_bridge_pane() {
    setup_main_env_pane "${WORKING_DIR}" "source /opt/ros/jazzy/setup.bash" "source ${WS_ROOT}/setup.bash"
    prepare_command "reset && python utilities/launch_rhc2rviz_bridge.py --ros2 --rhc_refs_in_h_frame \
--ns $SHM_NS --with_agent_refs --no_rhc_internal $(enabled "$PUB_HEIGHTMAP" && echo --show_heightmap)"
}

prepare_bag_dump_pane() {
    setup_main_env_pane "${WORKING_DIR}" "source /opt/ros/jazzy/setup.bash" "source ${WS_ROOT}/setup.bash"
    prepare_command "reset && python utilities/launch_periodic_bag_dump.py --ros2 --is_training --use_shared_drop_dir \
--pub_stime \
--ns $SHM_NS --rhc_refs_in_h_frame \
--bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG \
--srdf_path $SRDF_PATH_ROSBAG --with_agent_refs --no_rhc_internal $(enabled "$PUB_HEIGHTMAP" && echo --show_heightmap)"
}

add_isaac5x_tab() {
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

    cfg_key="${SHM_NS:-} ${RNAME:-} ${URDF_PATH:-} ${CLUSTER_CL_FNAME:-}"
    cfg_key="${cfg_key,,}"

    if [[ "$cfg_key" == *centauro* ]]; then
        RESOLVED_XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/centauro_ibrido.yaml"
    elif [[ "$cfg_key" == *b2w* ]]; then
        RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w/xbot2_basic.yaml"
    elif [[ "$cfg_key" == *kyon* ]]; then
        if [[ "$cfg_key" == *wheels_no_yaw* || "$cfg_key" == *no_yaw* ]]; then
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

add_rt_deployment_tab() {
    local rt_n_envs="${RT_N_ENVS:-1}"

    create_execution_layout

    go_to_pane 0
    resolve_xbot_config
    setup_main_env_pane "${WORKING_DIR}" \
        "source /opt/ros/jazzy/setup.bash" \
        "source /opt/xbot/setup.sh" \
        "source ${WS_ROOT}/setup.bash"
    build_world_cmd "aug_mpc_envs.world_interfaces.rt_deploy_world_interface" "$rt_n_envs" 0 0 0
    if [ -z "${RESOLVED_XBOT_CONFIG:-}" ]; then
        prepare_command "reset && echo 'XBOT_CONFIG is not set in the selected cfg'"
    else
        prepare_command "reset && set_xbot2_config \"${WS_ROOT}/src/${RESOLVED_XBOT_CONFIG}\" && (xbot2-core -S &) && python launch_world_interface.py $world_cmd"
    fi

    go_to_pane 1
    prepare_cluster_pane "$rt_n_envs"

    go_to_pane 2
    prepare_training_pane
}

add_xmj_tab() {
    local xmj_n_envs="${XMJ_N_ENVS:-1}"
    local xmj_headless="${XMJ_HEADLESS:-0}"
    local xmj_jnt_imp_config_path="$JNT_IMP_CF_PATH"
    local xmj_custom_args_names="$CUSTOM_ARGS_NAMES"
    local xmj_custom_args_dtype="$CUSTOM_ARGS_DTYPE"
    local xmj_custom_args_vals="$CUSTOM_ARGS_VALS"

    resolve_xbot_config
    resolve_xmj_files_dir

    if [ -n "${RESOLVED_XBOT_CONFIG:-}" ]; then
        xmj_jnt_imp_config_path="${WS_ROOT}/src/${RESOLVED_XBOT_CONFIG}"
    fi

    if [ -n "${RESOLVED_XMJ_FILES_DIR:-}" ] && [[ " ${xmj_custom_args_names} " != *" xmj_files_dir "* ]]; then
        xmj_custom_args_names+=" xmj_files_dir"
        xmj_custom_args_dtype+=" str"
        xmj_custom_args_vals+=" ${WS_ROOT}/src/${RESOLVED_XMJ_FILES_DIR}"
    fi

    create_execution_layout

    go_to_pane 0
    setup_main_env_pane "${WORKING_DIR}" \
        "source /opt/xbot/setup.sh" \
        "source ${WS_ROOT}/setup.bash"
    build_world_cmd "aug_mpc_envs.world_interfaces.xmj_world_interface" "$xmj_n_envs" "$xmj_headless" 1 0 \
        "$xmj_jnt_imp_config_path" "$xmj_custom_args_names" "$xmj_custom_args_dtype" "$xmj_custom_args_vals"
    prepare_command "reset && python launch_world_interface.py $world_cmd"

    go_to_pane 1
    prepare_cluster_pane "$xmj_n_envs"

    go_to_pane 2
    prepare_training_pane
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

    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10"
    prepare_mpcviz_pane "${WORKING_DIR_CENTAURO}" "$mpcviz_cmd"

    split_v
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --kyon_real --wheels"
    prepare_mpcviz_pane "${WORKING_DIR_QUAD}" "$mpcviz_cmd"

    split_h
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10 --b2w"
    prepare_mpcviz_pane "${WORKING_DIR_QUAD}" "$mpcviz_cmd"

    go_to_pane 0
    split_h
    mpcviz_cmd="python launch_mpcviz.py --ns $SHM_NS --nodes_perc 10"
    prepare_mpcviz_pane "${WORKING_DIR_TALOS}" "$mpcviz_cmd"
}

add_debug_tab() {
    prepare_gui_pane

    split_v
    prepare_ros_bridge_pane

    split_h
    prepare_bag_dump_pane
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
    -cfg|--cfg) config_file="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -f "$config_file" ]; then
    source "$config_file"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

echo "Will preload script cmds from $config_file"

N_FILES=${ULIM_N:-131072}

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

new_tab resources
add_resource_monitoring_tab

attach_to_session
