#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"

# # Define a cleanup function to send SIGINT to all child processes
cleanup() {
    echo "launch_training.sh: sending SIGINT to all child processes..."
    kill -INT $(jobs -p)  # Sends SIGINT to all child processes of the current script
}

# Trap EXIT, INT (Ctrl+C), and TERM signals to trigger cleanup
trap cleanup EXIT INT TERM

usage() {
  echo "Usage: $0
    [--cfg CFG] \
    [--unique_id UNIQUE_ID] \
    "
  exit 1
}

cfg_file_basepath="/root/ibrido_files"

# Default configuration file
config_file="${cfg_file_basepath}/training_cfg.sh"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --unique_id) unique_id="$2"; shift ;;
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

# clear tmp folder 
rm -r /tmp/*

# activate micromamba for this shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${MAMBA_ENV_NAME}

wandb login --relogin $WANDB_KEY # login to wandb

source /isaac-sim/setup_conda_env.sh
source $HOME/ibrido_ws/setup.bash

#!/bin/bash

# # Define the timestamp and log file based on RUN_NAME and unique id
log_file="${HOME}/ibrido_logs/ibrido_training_run_${RUN_NAME}_${unique_id}.log"
# # Redirect stdout and stderr to both the terminal and the log file
echo "launch_training.sh: logging output to $log_file"

exec > "$log_file" 2>&1

if (( set_ulim_eval )); then
  ulimit -n $ulim_n
fi

if (( $REMOTE_STEPPING )); then
  python $LRHC_DIR/launch_remote_env.py --headless --use_gpu --remote_stepping --robot_name $SHM_NS \
    --urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
    --use_custom_jnt_imp --jnt_imp_config_path $JNT_IMP_CF_PATH\
    --num_envs $N_ENVS --seed $SEED --timeout_ms $TIMEOUT_MS \
    --custom_args_names $CUSTOM_ARGS_NAMES \
    --custom_args_dtype $CUSTOM_ARGS_DTYPE \
    --custom_args_vals $CUSTOM_ARGS_VALS&
else
  python $LRHC_DIR/launch_remote_env.py --headless --use_gpu --robot_name $SHM_NS \
    --urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
    --use_custom_jnt_imp --jnt_imp_config_path $JNT_IMP_CF_PATH\
    --num_envs $N_ENVS --seed $SEED --timeout_ms $TIMEOUT_MS \
    --custom_args_names $CUSTOM_ARGS_NAMES \
    --custom_args_dtype $CUSTOM_ARGS_DTYPE \
    --custom_args_vals $CUSTOM_ARGS_VALS&
fi 

if (( $CLUSTER_DB )); then
  python $LRHC_DIR/launch_control_cluster.py --ns $SHM_NS --size $N_ENVS --timeout_ms $TIMEOUT_MS \
    --codegen_override_dir $CODEGEN_OVERRIDE_BDIR \
    --cloop \
    --enable_debug \
    --urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
    --custom_args_names $CUSTOM_ARGS_NAMES \
    --custom_args_dtype $CUSTOM_ARGS_DTYPE \
    --custom_args_vals $CUSTOM_ARGS_VALS&
else
  python $LRHC_DIR/launch_control_cluster.py --ns $SHM_NS --size $N_ENVS --timeout_ms $TIMEOUT_MS \
    --codegen_override_dir $CODEGEN_OVERRIDE_BDIR \
    --cloop \
    --urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
    --custom_args_names $CUSTOM_ARGS_NAMES \
    --custom_args_dtype $CUSTOM_ARGS_DTYPE \
    --custom_args_vals $CUSTOM_ARGS_VALS&
fi

if (( $REMOTE_STEPPING )); then
  if (( $OBS_NORM )); then
    python $LRHC_DIR/launch_train_env.py --ns $SHM_NS --run_name $RNAME --drop_dir $HOME/training_data --dump_checkpoints \
    --obs_norm --sac \
    --db --env_db --rmdb \
    --comment "$COMMENT" \
    --seed $SEED --timeout_ms $TIMEOUT_MS \
    --actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH\
    --critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH&
  else
    python $LRHC_DIR/launch_train_env.py --ns $SHM_NS --run_name $RNAME --drop_dir $HOME/training_data --dump_checkpoints \
    --sac \
    --db --env_db --rmdb \
    --comment "$COMMENT" \
    --seed $SEED --timeout_ms $TIMEOUT_MS \
    --actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH\
    --critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH&
  fi 
  
fi 

if (( $LAUNCH_ROSBAG && $CLUSTER_DB)); then
  source /opt/ros/humble/setup.bash
  if (( $REMOTE_STEPPING )); then
    python $LRHC_DIR/launch_periodic_bag_dump.py --ros2 --use_shared_drop_dir \
      --ns $SHM_NS --rhc_refs_in_h_frame \
      --srdf_path $SRDF_PATH_ROSBAG \
      --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG --with_agent_refs &
  else
    python $LRHC_DIR/launch_periodic_bag_dump.py --ros2 --use_shared_drop_dir \
      --ns $SHM_NS --rhc_refs_in_h_frame \
      --srdf_path $SRDF_PATH_ROSBAG \
      --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG &
  fi
fi

wait # wait for all to exit