#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_u24/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--cfg <config_file>] [--wdb_key <wandb_key>] [--ros_global] [--run_token <token>] [--set VAR=VALUE] [--dry-run]"
    exit 1
}

# Function to handle cleanup on exit
cleanup() {
    if [ -z "${training_script_pid:-}" ]; then
        echo "execute.sh: No training PID recorded, nothing to clean up."
        return
    fi

    echo "execute.sh: Cleaning up and sending SIGINT to training process..."
    # Send SIGINT to the process group to propagate to children; fall back to the PID
    kill -SIGINT -"$training_script_pid" 2>/dev/null || kill -SIGINT "$training_script_pid" 2>/dev/null
    # Loop until the process is no longer found in `ps` output
    while ps -p "$training_script_pid" > /dev/null 2>&1; do
        echo "execute.sh: training script still alive."
        sleep 1  # Check every second
    done
    echo "execute.sh: training script exited."
}

# Trap and forward signals to singularity process for clean exit
trap cleanup SIGINT SIGTERM

use_sudo=false # whether to use superuser privileges
wandb_key_default="$WANDB_KEY" # Use the existing WANDB_KEY by default
ros_localhost_only=1
run_token=""
dry_run=0
cfg_overrides=()

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -cfg|--cfg) config_file="$2"; shift ;; # Set custom config file if provided
        -wdb_key|--wdb_key) wandb_key="$2"; shift ;; # Override WANDB_KEY if provided
        --ros_global) ros_localhost_only=0 ;;
        --run_token) run_token="$2"; shift ;; # optional unique token for process identification
        --set) cfg_overrides+=("$2"); shift ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Use default WANDB_KEY if no --wdb_key argument was provided
wandb_key="${wandb_key:-$wandb_key_default}"

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

training_script="launch_training.sh"

# Generate a unique ID based on the current timestamp
job_id=$(echo "$SCHED_JOBID" | cut -d'.' -f1)
unique_id="_$(date +%Y_%m_%d_%H_%M_%S)_ID${job_id}" # just used to retrive process ID

training_cmd="$training_script --unique_id ${unique_id} --cfg $config_file"
for cfg_override in "${cfg_overrides[@]}"; do
    cfg_override_name="${cfg_override%%=*}"
    if [[ "$cfg_override" != *=* || ! "$cfg_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid --set override: $cfg_override"
        exit 1
    fi
    printf -v cfg_override_quoted '%q' "$cfg_override"
    training_cmd+=" --set ${cfg_override_quoted}"
done
if (( dry_run )); then
    training_cmd+=" --dry-run"
fi

singularity_cmd="singularity exec \
    --env \"WANDB_KEY=$wandb_key\"\
    --env \"ROS_LOCALHOST_ONLY=$ros_localhost_only\"\
    --env \"BYOBU_CONFIG_DIR=/root/.byobu\"\
    --bind $binddirs\
    --no-mount home,cwd \
    --pwd /root \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_cmd \
    "

# Run the singularity command and get its PID
# Launch the training inside its own session so we can signal the whole group
if $use_sudo; then
    sudo setsid bash -c "$singularity_cmd" &
else
    setsid bash -c "$singularity_cmd" &
fi

# Capture the PID of the sudo or bash process
parent_pid=$!

if (( dry_run )); then
    wait "$parent_pid"
    exit $?
fi

# Use pgrep to find the actual singularity process ID associated with the command
# Give it a second to start to ensure the process is up and running
sleep 3
training_script_pid=$(pgrep -f "$training_cmd" | head -n 1)

# Check if the singularity PID was found and print it for debugging or logging
if [ -n "$training_script_pid" ]; then
    echo "execute.sh: Training process PID: $training_script_pid"
else
    # Fallback to the parent pid so cleanup still works
    training_script_pid=$parent_pid
    echo "execute.sh: Failed to locate training process PID, falling back to parent pid ${parent_pid}."
fi

wait "$parent_pid"
