#!/bin/bash
# set -e # exiting if any cmd fails

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"
source "${IBRIDO_CONTAINERS_PREFIX}/files/training_cfg.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--set_ulim|-ulim] [--wandb_key|-w <key>] [--comment|-c <key>]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
wandb_key=""
comment=""
set_ulim=false

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
        -ulim|--set_ulim) set_ulim=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

training_script="/root/ibrido_ws/launch_training.sh"
# training_script="launch_training.sh"

if $use_sudo; then
    if $set_ulim; then
        sudo singularity exec \
            --env "WANDB_KEY=$wandb_key"\
            -B /tmp/.X11-unix:/tmp/.X11-unix\
            -B /etc/localtime:/etc/localtime:ro \
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --set_ulim \
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $comment \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    else
        sudo singularity exec \
            --env "WANDB_KEY=$wandb_key"\
            -B /tmp/.X11-unix:/tmp/.X11-unix\
            -B /etc/localtime:/etc/localtime:ro \
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $comment \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    fi
else
    if $set_ulim; then
        singularity exec \
            --env "WANDB_KEY=$wandb_key"\
            -B /tmp/.X11-unix:/tmp/.X11-unix\
            -B /etc/localtime:/etc/localtime:ro \
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --set_ulim\
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $comment \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    else
        singularity exec \
            --env "WANDB_KEY=$wandb_key"\
            -B /tmp/.X11-unix:/tmp/.X11-unix\
            -B /etc/localtime:/etc/localtime:ro \
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $comment \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    fi
fi

