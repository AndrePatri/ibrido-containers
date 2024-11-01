#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Default configuration file
config_file="${IBRIDO_CONTAINERS_PREFIX}/files/training_cfg.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--cfg <config_file>] [--wdb_key <wandb_key>]"
    exit 1
}

# Function to handle cleanup on exit
cleanup() {
    echo "execute.sh: Cleaning up and sending SIGINT to training process..."
    kill -SIGINT "$training_pid"  # Send SIGINT to singularity process
    wait "$training_pid"  # Wait for process to exit
    echo "execute.sh: training script exited."
}

# Trap and forward signals to singularity process for clean exit
trap cleanup SIGINT SIGTERM

use_sudo=false # whether to use superuser privileges
wandb_key_default="$WANDB_KEY" # Use the existing WANDB_KEY by default

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -cfg|--cfg) config_file="$2"; shift ;; # Set custom config file if provided
        -wdb_key|--wdb_key) wandb_key="$2"; shift ;; # Override WANDB_KEY if provided
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Use default WANDB_KEY if no --wdb_key argument was provided
wandb_key="${wandb_key:-$wandb_key_default}"

# Source the configuration file
if [ -f "$config_file" ]; then
    source "$config_file"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

training_script="launch_training.sh"

# Generate a unique ID based on the current timestamp
unique_id="ibrido_run_$(date +%Y%m%d%H%M%S)" # just used to retrive process ID

singularity_cmd="singularity exec \
    --env \"WANDB_KEY=$wandb_key\"\
    --env \"ROS_LOCALHOST_ONLY=1\"\
    --bind $binddirs\
    --no-mount home,cwd \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script --unique_id ${unique_id} \
    --urdf_path "\$URDF_PATH" \
    --srdf_path "\$SRDF_PATH" \
    --jnt_imp_config_path "\$JNT_IMP_CF_PATH" \
    --cluster_client_fname $CLUSTER_CL_FNAME \
    --num_envs $N_ENVS \
    --set_ulim $SET_ULIM\
    --ulim_n $ULIM_N \
    --ns $SHM_NS \
    --run_name $RNAME \
    --comment \"$COMMENT\" \
    --seed $SEED \
    --timeout_ms $TIMEOUT_MS \
    --codegen_override "\$CODEGEN_OVERRIDE_BDIR" \
    --launch_rosbag $LAUNCH_ROSBAG \
    --bag_sdt $BAG_SDT \
    --ros_bridge_dt $BRIDGE_DT \
    --dump_dt_min $DUMP_DT \
    --env_idx_bag $ENV_IDX_BAG \
    --custom_args_names \"$CUSTOM_ARGS_NAMES\" \
    --custom_args_dtype \"$CUSTOM_ARGS_DTYPE\" \
    --custom_args_vals \"$CUSTOM_ARGS_VALS\" \
    --remote_stepping $REMOTE_STEPPING \
    --cluster_db $CLUSTER_DB \
    "

# Run the singularity command and get its PID
if $use_sudo; then
    sudo bash -c "$singularity_cmd" &
else
    bash -c "$singularity_cmd" &
fi

# Capture the PID of the sudo or bash process
parent_pid=$!

# Use pgrep to find the actual singularity process ID associated with the command
# Give it a second to start to ensure the process is up and running
sleep 3
training_pid=$(pgrep -f "$training_script --unique_id ${unique_id}")

# Check if the singularity PID was found and print it for debugging or logging
if [ -n "$training_pid" ]; then
    echo "execute.sh: Training process PID: $training_pid"
else
    echo "execute.sh: Failed to locate training process PID."
fi

wait