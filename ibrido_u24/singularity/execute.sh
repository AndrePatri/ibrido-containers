#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_u24/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] (--cfg <run_cfg> | --bundle <bundle_dir_or_bundle_yaml> --cfg <transfer_cfg>) [--wandb_key|--wdb_key|-w <wandb_key>] [--ros_global] [--run_token <token>] [--set VAR=VALUE] [--allow_contract_override] [--dry-run]"
    exit 1
}

# Forward the first Ctrl-C gracefully; later interrupts explicitly escalate.
interrupt_count=0
parent_pid=""

handle_interrupt() {
    if [ -z "$parent_pid" ] || ! kill -0 "$parent_pid" 2>/dev/null; then
        return
    fi

    interrupt_count=$((interrupt_count + 1))
    if (( interrupt_count == 1 )); then
        echo "execute.sh: requesting graceful shutdown; press Ctrl-C again to force it..."
        kill -SIGINT "$parent_pid" 2>/dev/null || true
    elif (( interrupt_count == 2 )); then
        echo "execute.sh: forcing shutdown with SIGTERM..."
        kill -SIGTERM -"$parent_pid" 2>/dev/null || kill -SIGTERM "$parent_pid" 2>/dev/null || true
    else
        echo "execute.sh: forcing shutdown with SIGKILL..."
        kill -SIGKILL -"$parent_pid" 2>/dev/null || kill -SIGKILL "$parent_pid" 2>/dev/null || true
    fi
}

handle_termination() {
    if [ -n "$parent_pid" ] && kill -0 "$parent_pid" 2>/dev/null; then
        interrupt_count=2
        echo "execute.sh: received SIGTERM; forwarding it to the training process group..."
        kill -SIGTERM -"$parent_pid" 2>/dev/null || kill -SIGTERM "$parent_pid" 2>/dev/null || true
    fi
}

trap handle_interrupt SIGINT
trap handle_termination SIGTERM

use_sudo=false # whether to use superuser privileges
wandb_key_default="${WANDB_KEY:-${WANDB_API_KEY:-}}" # Use the existing WANDB_KEY/WANDB_API_KEY by default
ros_localhost_only=1
run_token=""
dry_run=0
bundle_path=""
allow_contract_override=0
cfg_overrides=()

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -cfg|--cfg) config_file="$2"; shift ;; # Set custom config file if provided
        --bundle) bundle_path="$2"; shift ;;
        -w|--wandb_key|--wandb-key|-wdb_key|--wdb_key) wandb_key="$2"; shift ;; # Override WANDB_KEY if provided
        --ros_global) ros_localhost_only=0 ;;
        --run_token) run_token="$2"; shift ;; # optional unique token for process identification
        --set) cfg_overrides+=("$2"); shift ;;
        --allow_contract_override) allow_contract_override=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Use default WANDB_KEY/WANDB_API_KEY if no explicit argument was provided
wandb_key="${wandb_key:-$wandb_key_default}"
if [ -n "$wandb_key" ] && [ "${#wandb_key}" -lt 40 ]; then
    echo "execute.sh: WANDB key looks invalid: expected at least 40 characters, got ${#wandb_key}."
    exit 1
fi

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

# Generate a unique ID based on the current timestamp
job_id=$(echo "$SCHED_JOBID" | cut -d'.' -f1)
unique_id="_$(date +%Y_%m_%d_%H_%M_%S)_ID${job_id}" # just used to retrive process ID

if [ -z "${bundle_path:-}" ] && [ -z "${config_file:-}" ]; then
    echo "execute.sh: --cfg is required"
    usage
fi

if [ -n "${bundle_path:-}" ] && [ -z "${config_file:-}" ]; then
    echo "execute.sh: --cfg <transfer_cfg> is required with --bundle"
    usage
fi

if [ -n "${bundle_path:-}" ]; then
    training_script="launch_bundle.sh"
    container_bundle_path="$bundle_path"
    if [[ -n "${IBRIDO_TRAINING_DATA:-}" && "$container_bundle_path" == "${IBRIDO_TRAINING_DATA}"* ]]; then
        container_bundle_path="/root/training_data${container_bundle_path#"${IBRIDO_TRAINING_DATA}"}"
    fi
    printf -v container_bundle_path_quoted '%q' "$container_bundle_path"
    printf -v config_file_quoted '%q' "$config_file"
    training_cmd="$training_script --unique_id ${unique_id} --bundle ${container_bundle_path_quoted} --cfg ${config_file_quoted}"
    if (( allow_contract_override )); then
        training_cmd+=" --allow_contract_override"
    fi
else
    training_script="launch_training.sh"
    training_cmd="$training_script --unique_id ${unique_id}"
    if [ -n "${config_file:-}" ]; then
        printf -v config_file_quoted '%q' "$config_file"
        training_cmd+=" --cfg ${config_file_quoted}"
    fi
fi

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
    --env \"WANDB_API_KEY=$wandb_key\"\
    --env \"ROS_LOCALHOST_ONLY=$ros_localhost_only\"\
    --env \"BYOBU_CONFIG_DIR=/root/.byobu\"\
    --bind $binddirs\
    --no-mount home,cwd \
    --pwd /root \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido.sif $training_cmd \
    "

# Run the singularity command and get its PID
# Launch the training inside its own session so we can signal the whole group
if $use_sudo; then
    setsid sudo bash -c "exec $singularity_cmd" &
else
    setsid bash -c "exec $singularity_cmd" &
fi

# This PID is also the process-group ID because setsid starts a new session.
parent_pid=$!

if (( dry_run )); then
    wait "$parent_pid"
    exit $?
fi

echo "execute.sh: Training process-group leader PID: $parent_pid"
training_status=0
while kill -0 "$parent_pid" 2>/dev/null; do
    wait "$parent_pid"
    training_status=$?
done

trap - SIGINT SIGTERM
exit "$training_status"
