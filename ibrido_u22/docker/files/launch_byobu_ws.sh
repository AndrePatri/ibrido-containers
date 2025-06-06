#!/bin/bash

# Set Italian keyboard layout
export XMODIFIERS=@im=ibus
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus

SLEEP_FOR=0.1
BYOBU_WS_NAME="ibrido_ws"
WS_ROOT="$HOME/ibrido_ws"
WORKING_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"
WORKING_DIR2="$WS_ROOT/src/KyonRLStepping/kyonrlstepping/scripts"

MAMBAENVNAME="ibrido"
N_FILES=28672 # to allow more open files (for semaphores/mutexes etc..)

# Array of directories
directories=(
    "$WS_ROOT/src/LRHControl"
    "$WS_ROOT/src/CoClusterBridge"
    "$WS_ROOT/src/LRHControlEnvs"
    "$WS_ROOT/src/EigenIPC"
    "$WS_ROOT/src/horizon"
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

    execute_command "mamba activate ${MAMBAENVNAME}"

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

# launch terminator window
byobu kill-session -t ${BYOBU_WS_NAME}

byobu new-session -d -s ${BYOBU_WS_NAME} -c ${WORKING_DIR} -n ${BYOBU_WS_NAME} # -d "detached" session

# tab 0
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
#execute_command "source ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_conda_env.sh"
execute_command "source /isaac-sim/setup_conda_env.sh"
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally 
clear_terminal
prepare_command "reset && python launch_sim_env.py --headless --remote_stepping --robot_name {} --robot_pkg_name {} --num_envs {}"

split_v
execute_command "cd ${WORKING_DIR2}"
activate_mamba_env
execute_command "source $WS_ROOT/setup.bash"
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_control_cluster.py --enable_debug --ns {} --size {}"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_GUI.py --ns {}"

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_keyboard_cmds.py --ns {}"

go_to_pane 0 

split_h
execute_command "cd ${WORKING_DIR}"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_train_env.py --ns {} --run_name {} --drop_dir $HOME/training_data --dump_checkpoints --comment "" "

split_h
execute_command "cd ${WORKING_DIR}"
# execute_command "source /opt/ros/noetic/setup.bash"
execute_command "source /opt/ros/humble/setup.bash"
execute_command "source $WS_ROOT/setup.bash"
activate_mamba_env
increase_file_limits_locally
clear_terminal
prepare_command "reset && python launch_rhc2ros_bridge.py --ros2 --with_agent_refs --ns {}"

# tab 1
new_tab
execute_command "cd ${WORKING_DIR}"

split_h
execute_command "cd ${WORKING_DIR2}"
# execute_command "source /opt/ros/noetic/setup.bash"
activate_mamba_env
execute_command "source /opt/ros/humble/setup.bash"
clear_terminal
prepare_command "reset && python launch_rhcviz.py --ns {}"

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