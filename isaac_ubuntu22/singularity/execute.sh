#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${SCRIPT_DIR}/files/bind_list.sh"
source "${SCRIPT_DIR}/files/training_cfg.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--wandb_key|-w <key>] [--comment|-c <key>]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
wandb_key=""
comment=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -w|--wandb_key) 
            if [[ -n "$2" && "$2" != "-"* ]]; then
                wandb_key=$2
                shift
            else
                echo "Error: --wandb_key requires a non-empty argument."
                usage
            fi
            ;;
        -c|--comment) 
            if [[ -n "$2" && "$2" != "-"* ]]; then
                comment=$2
                shift
            else
                echo "Error: --comment requires a non-empty argument."
                usage
            fi
            ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

if $use_sudo; then
    sudo singularity exec \
        --env "WANDB_KEY=$wandb_key"\
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_isaac.sif launch_training.sh \
            --robot_name $RB_NAME \
            --robot_pkg_name $RB_PNAME \
            --num_envs $N_ENVS \
            --ulim_n $ULIM_N \
            --ns $SHM_NS \
            --run_name $RNAME \
            --comment $comment 
else
    singularity exec \
        --env "WANDB_KEY=$wandb_key"\
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_isaac.sif launch_training.sh \
            --robot_name $RB_NAME \
            --robot_pkg_name $RB_PNAME \
            --num_envs $N_ENVS \
            --ulim_n $ULIM_N \
            --ns $SHM_NS \
            --run_name $RNAME \
            --comment $comment 
fi

