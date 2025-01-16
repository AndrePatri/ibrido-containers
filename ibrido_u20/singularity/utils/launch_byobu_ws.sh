#!/bin/bash

# Set Italian keyboard layout
export XMODIFIERS=@im=ibus
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus

SLEEP_FOR=0.1
BYOBU_WS_NAME="ibrido_xbot"
WS_ROOT="$HOME/ibrido_ws"
WORKING_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"

MAMBAENVNAME="${MAMBA_ENV_NAME}"
N_FILES=28672 # to allow more open files (for semaphores/mutexes etc..)

# Array of directories
directories=(
    "$WS_ROOT/src/KyonRLStepping"
    "$WS_ROOT/src/LRHControl"
    "$WS_ROOT/src/CoClusterBridge"
    "$WS_ROOT/src/LRHControlEnvs"
    # Add more directories as needed
)

press_enter() {

    byobu send-keys Enter
    sleep $SLEEP_FOR
}

# Function to execute common commands in Terminator terminal
execute_command() {
    byobu send-keys "$1"
    press_enter
    sleep $SLEEP_FOR
}

prepare_command() {
    byobu send-keys "$1"
    sleep $SLEEP_FOR
}

go_to_pane() {

    byobu select-pane -t $1

}

go_to_window() {

    byobu select-window -t $1

}

attach_to_session() {

    byobu attach-session -t ${BYOBU_WS_NAME} 

}

activate_mamba_env() {

    execute_command "eval \"\$(micromamba shell hook --shell bash)\""
    execute_command "micromamba activate ${MAMBAENVNAME}"

}

clear_terminal() {

    execute_command "clear"

}

increase_file_limits_locally() {

    # for shared memory

    execute_command "ulimit -n ${N_FILES}"

}

split_h() {

    byobu split-window -p 50 -v

}

split_v() {

    byobu split-window -p 50 -h

}

new_tab() {

    byobu new-window

}

# Function to navigate to a directory and split Terminator horizontally
cd_and_split() {

    execute_command "cd $1"
    
    # Check if it's the last directory before splitting
    if [ "$1" != "${directories[-1]}" ]; then
    
        split_h

    fi
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -cfg|--cfg) config_file="${cfg_file_basepath}/$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Source the configuration file
if [ -f "$config_file" ]; then
    source "$config_file"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

echo "Will preload script cmds from $config_file"

# clear tmp folder 
# rm -r /tmp/*

# launch terminator window
byobu kill-session -t ${BYOBU_WS_NAME}

byobu new-session -d -s ${BYOBU_WS_NAME} -c ${WORKING_DIR} -n ${BYOBU_WS_NAME} # -d "detached" session

# tab 0
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
#execute_command "source ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_conda_env.sh"
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/xbot/setup.sh"
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally 
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
remote_env_cmd="--headless --robot_name $SHM_NS \
--urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
--use_custom_jnt_imp --jnt_imp_config_path $JNT_IMP_CF_PATH \
--env_fname $REMOTE_ENV_FNAME \
--cluster_dt $CLUSTER_DT \
--num_envs $N_ENVS --seed $SEED --timeout_ms $TIMEOUT_MS \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--enable_debug "
if (( $REMOTE_STEPPING )); then
remote_env_cmd+="--remote_stepping "
fi 
prepare_command "reset && python launch_remote_env.py $remote_env_cmd "

split_v
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally
clear_terminal
cluster_cmd="--ns $SHM_NS --size $N_ENVS --timeout_ms $TIMEOUT_MS \
--urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--cluster_dt $CLUSTER_DT \
--n_nodes $N_NODES "

if (( $CLUSTER_DB )); then
cluster_cmd+="--enable_debug "
fi
if (( $IS_CLOSED_LOOP )); then
cluster_cmd+="--cloop "
fi
prepare_command "reset && python launch_control_cluster.py $cluster_cmd"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/xbot/setup.sh"
increase_file_limits_locally
execute_command "set_xbot2_config $HOME/ibrido_ws/src/$XBOT_CONFIG"
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && xbot2-core -S"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_GUI.py --ns $SHM_NS"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_rhc_keybrd_cmds.py --ns $SHM_NS"

split_h
execute_command "cd $WORKING_DIR"
activate_mamba_env
increase_file_limits_locally
prepare_command "reset && python launch_agent_keybrd_cmds.py --ns $SHM_NS"

go_to_pane 0 

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
training_env_cmd="--dump_checkpoints --ns $SHM_NS --drop_dir $HOME/training_data \
--sac --db --env_db --rmdb \
--seed $SEED --timeout_ms $TIMEOUT_MS \
--env_fname $TRAIN_ENV_FNAME --env_classname $TRAIN_ENV_CNAME \
--demo_stop_thresh $DEMO_STOP_THRESH  \
--actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH \
--critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH \
--compression_ratio $COMPRESSION_RATIO "
if (( $OBS_NORM )); then
training_env_cmd+="--obs_norm "
fi
if (( $OBS_RESCALING )); then
training_env_cmd+="--obs_rescale "
fi
if (( $WEIGHT_NORM )); then
training_env_cmd+="--add_weight_norm "
fi
prepare_command "reset && python launch_train_env.py $training_env_cmd --comment \"$COMMENT\""

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
#execute_command "source ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_conda_env.sh"
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/xbot/setup.sh"
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally 
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal

split_h
execute_command "cd ${WORKING_DIR}"
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source $WS_ROOT/setup.bash"
activate_mamba_env
increase_file_limits_locally
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && python launch_rhc2ros_bridge.py --rhc_refs_in_h_frame --ns $SHM_NS --with_agent_refs "

split_h
execute_command "cd ${WORKING_DIR}"
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source $WS_ROOT/setup.bash"
activate_mamba_env
increase_file_limits_locally
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && python launch_periodic_bag_dump.py --is_training --use_shared_drop_dir \
--ns $SHM_NS --rhc_refs_in_h_frame \
--bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG \
--srdf_path $SRDF_PATH_ROSBAG --with_agent_refs \
--pub_stime \
--xbot "

# tab 1
new_tab
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source /opt/ros/noetic/setup.bash"
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && ./replay_bag.bash $HOME/training_data/{}"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source /opt/ros/noetic/setup.bash"
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && python launch_rhcviz.py --ns $SHM_NS --nodes_perc 10"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/xbot/setup.sh"
clear_terminal
prepare_command "reset && xbot2-gui"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
execute_command "source /opt/ros/noetic/setup.bash"
export ROS_MASTER_URI=$ROS_MASTER_URI
export ROS_IP=$ROS_IP
clear_terminal
prepare_command "reset && roscore"

# tab2
new_tab
execute_command "htop"

split_h
execute_command "cd ${WORKING_DIR}"
execute_command "nvtop"
clear_terminal

# tab 3
new_tab

# Loop through directories and navigate to each one
for dir in "${directories[@]}"; do
    cd_and_split "$dir"
done

# we attach to the detached session
attach_to_session