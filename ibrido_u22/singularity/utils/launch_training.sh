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

source /isaac-sim/setup_conda_env.sh
source $HOME/ibrido_ws/setup.bash

#!/bin/bash

# Define the timestamp and log file based on RUN_NAME and unique id
base_log_dir="${HOME}/ibrido_logs/ibrido_run_${unique_id}"
mkdir -p "$base_log_dir"
log_remote="${base_log_dir}/ibrido_remote_env_${RUN_NAME}_${unique_id}.log"
log_cluster="${base_log_dir}/ibrido_rhc_cluster_${RUN_NAME}_${unique_id}.log"
log_train="${base_log_dir}/ibrido_train_env_${RUN_NAME}_${unique_id}.log"
log_bag="${base_log_dir}/ibrido_rosbag_env_${RUN_NAME}_${unique_id}.log"

# Ensure the log directory exists

# # Redirect stdout and stderr to both the terminal and the log file
echo "
launch_training.sh: logging output to->
remote env: $log_remote
rhc cluster: $log_cluster
train. env: $log_train
log bag: $log_bag
"

if (( $SET_ULIM )); then
  ulimit -n $ULIM_N
fi

# remote env
remote_env_cmd="--headless --use_gpu  --robot_name $SHM_NS \
--urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
--use_custom_jnt_imp --jnt_imp_config_path $JNT_IMP_CF_PATH \
--cluster_dt $CLUSTER_DT \
--num_envs $N_ENVS --seed $SEED --timeout_ms $TIMEOUT_MS \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS "
if (( $REMOTE_STEPPING )); then
remote_env_cmd+="--remote_stepping "
fi 
python $LRHC_DIR/launch_remote_env.py $remote_env_cmd > "$log_remote" 2>&1 &

# cluster
cluster_cmd="--ns $SHM_NS --size $N_ENVS --timeout_ms $TIMEOUT_MS \
--codegen_override_dir $CODEGEN_OVERRIDE_BDIR \
--cloop \
--urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS "
if (( $CLUSTER_DB )); then
cluster_cmd+="--enable_debug "
fi
python $LRHC_DIR/launch_control_cluster.py $cluster_cmd > "$log_cluster" 2>&1 &

# train env
if (( $REMOTE_STEPPING )); then
training_env_cmd="--dump_checkpoints --ns $SHM_NS --run_name $RNAME --drop_dir $HOME/training_data \
--sac --db --env_db --rmdb \
--seed $SEED --timeout_ms $TIMEOUT_MS \
--env_fname linvel_env_baseline --env_classname LinVelTrackBaseline \
--actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH \
--critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH "
if (( $OBS_NORM )); then
training_env_cmd+="--obs_norm "
fi
if (( $OBS_RESCALING )); then
training_env_cmd+="--obs_rescale "
fi

wandb login $WANDB_KEY # login to wandb

python $LRHC_DIR/launch_train_env.py $training_env_cmd --comment "\"$COMMENT\"" > "$log_train" 2>&1 &
fi

# rosbag db
if (( $ENV_IDX_BAG > 0 && $CLUSTER_DB)); then
  source /opt/ros/humble/setup.bash
  rosbag_cmd="--ros2 --use_shared_drop_dir --pub_stime\
  --ns $SHM_NS --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG "
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs "
  fi
  python $LRHC_DIR/launch_periodic_bag_dump.py $rosbag_cmd > "$log_bag" 2>&1 &
fi

# demo env db
sleep 2 # wait a bit
if (( $ENV_IDX_BAG_DEMO > 0 && $CLUSTER_DB)); then
  source /opt/ros/humble/setup.bash
  rosbag_cmd="--ros2 --use_shared_drop_dir\
  --ns $SHM_NS --remap_ns "${SHM_NS}_demo" \
  --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG_DEMO "
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs "
  fi
  python $LRHC_DIR/launch_periodic_bag_dump.py $rosbag_cmd > "$log_bag" 2>&1 &
fi

# expl env db
sleep 2 # wait a bit
if (( $ENV_IDX_BAG_EXPL > 0 && $CLUSTER_DB)); then
  source /opt/ros/humble/setup.bash
  rosbag_cmd="--ros2 --use_shared_drop_dir\
  --ns $SHM_NS --remap_ns "${SHM_NS}_expl" \
  --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG_EXPL "
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs "
  fi
  python $LRHC_DIR/launch_periodic_bag_dump.py $rosbag_cmd > "$log_bag" 2>&1 &
fi

wait # wait for all to exit