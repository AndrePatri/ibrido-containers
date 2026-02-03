#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--cfg <config_file>] [--wdb_key <wandb_key>] [--no_unique_id"
    exit 1
}

# Function to handle cleanup on exit
cleanup() {
    echo "execute.sh: Cleaning up and sending SIGINT to training process..."
    kill -SIGINT "$training_script_pid"  # Send SIGINT to singularity process
    # Loop until the process is no longer found in `ps` output
    while ps -p "$training_script_pid" > /dev/null; do
        echo "execute.sh: training script still alive."
        sleep 2  # Check every second
    done
    echo "execute.sh: training script exited."
}

# Trap and forward signals to singularity process for clean exit
trap cleanup SIGINT SIGTERM

use_sudo=false # whether to use superuser privileges
wandb_key_default="$WANDB_KEY" # Use the existing WANDB_KEY by default
no_unique_id=0

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -cfg|--cfg) config_file="$2"; shift ;; # Set custom config file if provided
        -wdb_key|--wdb_key) wandb_key="$2"; shift ;; # Override WANDB_KEY if provided
        --no_unique_id) no_unique_id=1 ;; # do not append unique id to the run
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
job_id=$(echo "$SCHEDULER_JOBID" | cut -d'.' -f1)
if [ $no_unique_id -eq 1 ]; then
    unique_id=""
else
    unique_id="_$(date +%Y_%m_%d_%H_%M_%S)_ID${job_id}" # just used to retrive process ID
fi

training_cmd="$training_script --unique_id ${unique_id} --cfg $config_file"

singularity_cmd="singularity exec \
    --cleanenv \
    --env \"WANDB_KEY=$wandb_key\"\
    --bind $binddirs \
    --no-mount home,cwd \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif $training_cmd \
    "

# Run the singularity command and get its PID
if $use_sudo; then
    singularity_cmd+="--fakeroot --net "
fi

bash -c "$singularity_cmd" &

# Capture the PID of the sudo or bash process
parent_pid=$!

# Use pgrep to find the actual singularity process ID associated with the command
# Give it a second to start to ensure the process is up and running
sleep 3
training_script_pid=$(pgrep -f "$training_cmd")

# Check if the singularity PID was found and print it for debugging or logging
if [ -n "$training_script_pid" ]; then
    echo "execute.sh: Training process PID: $training_script_pid"
else
    echo "execute.sh: Failed to locate training process PID."
fi

wait

