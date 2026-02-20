#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit 1
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

usage() {
    echo "Usage: $0 [--use_sudo|-s] [--cfg <config_file>] [--wdb_key <wandb_key>] [--no_unique_id]"
    exit 1
}

use_sudo=false
wandb_key_default="${WANDB_KEY:-}"
no_unique_id=0
config_file=""
wandb_key=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -cfg|--cfg)
            if [[ -z "${2:-}" ]]; then
                echo "Missing value for --cfg"
                usage
            fi
            config_file="$2"; shift ;;
        -wdb_key|--wdb_key)
            if [[ -z "${2:-}" ]]; then
                echo "Missing value for --wdb_key"
                usage
            fi
            wandb_key="$2"; shift ;;
        --no_unique_id) no_unique_id=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

wandb_key="${wandb_key:-$wandb_key_default}"

# convert bind dirs to comma-separated list
IFS=','
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS

training_script="launch_training.sh"

job_id="$(echo "${SCHEDULER_JOBID:-}" | cut -d'.' -f1)"
if [ "$no_unique_id" -eq 1 ]; then
    unique_id=""
else
    unique_id="_$(date +%Y_%m_%d_%H_%M_%S)_ID${job_id}"
fi

training_cmd="$training_script"
if [[ -n "$unique_id" ]]; then
    training_cmd+=" --unique_id ${unique_id}"
fi
if [[ -n "$config_file" ]]; then
    training_cmd+=" --cfg $config_file"
fi

singularity_cmd="singularity exec \
    --cleanenv \
    --env \"WANDB_KEY=$wandb_key\" \
    --bind $binddirs \
    --no-mount home,cwd \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif $training_cmd \
"

if $use_sudo; then
    singularity_cmd+=" --fakeroot --net"
fi

# -------------------------
# Signal handling (escalate only on 2nd+ signal)
# -------------------------
_in_cleanup=0
training_script_pid=""

cleanup_graceful() {
    echo "execute.sh: graceful cleanup: sending SIGINT to training process (if running)..."

    if [[ -n "${training_script_pid:-}" ]] && kill -0 "$training_script_pid" 2>/dev/null; then
        echo "execute.sh: sending SIGINT to PID $training_script_pid"
        kill -INT "$training_script_pid" 2>/dev/null || true

        # Wait indefinitely until it exits (no timeout)
        while kill -0 "$training_script_pid" 2>/dev/null; do
            echo "execute.sh: training script still alive..."
            sleep 2
        done

        echo "execute.sh: training script exited."
    else
        echo "execute.sh: no training_script_pid to signal (or it's not running)."
    fi
}

cleanup_hard() {
    echo "execute.sh: HARD cleanup: sending SIGKILL to training process (if running)..."
    if [[ -n "${training_script_pid:-}" ]] && kill -0 "$training_script_pid" 2>/dev/null; then
        echo "execute.sh: sending SIGKILL to PID $training_script_pid"
        kill -KILL "$training_script_pid" 2>/dev/null || true
    else
        echo "execute.sh: no training_script_pid to kill (or it's not running)."
    fi
}

on_signal() {
    local sig="$1"

    if [ "$_in_cleanup" -eq 1 ]; then
        echo "execute.sh: received $sig again during cleanup -> escalating"
        cleanup_hard
        exit 128
    fi

    _in_cleanup=1
    echo "execute.sh: received $sig -> starting graceful cleanup"
    cleanup_graceful

    case "$sig" in
        INT)  exit 130 ;;
        TERM) exit 143 ;;
        QUIT) exit 131 ;;
        *)    exit 1 ;;
    esac
}

on_exit() {
    if [ "$_in_cleanup" -eq 0 ]; then
        _in_cleanup=1
        cleanup_graceful
    fi
}

trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal QUIT' QUIT
trap 'on_exit' EXIT
# NOTE: SIGKILL cannot be trapped.

# -------------------------
# Launch
# -------------------------
bash -c "$singularity_cmd" &
parent_pid=$!

sleep 3
training_script_pid="$(pgrep -f "$training_cmd" | head -n 1 || true)"

if [[ -n "$training_script_pid" ]]; then
    echo "execute.sh: Training process PID: $training_script_pid"
else
    echo "execute.sh: Failed to locate training process PID."
fi

wait
