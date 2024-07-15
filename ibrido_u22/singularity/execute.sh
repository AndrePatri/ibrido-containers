#!/bin/bash
# set -e # exiting if any cmd fails

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"
source "${IBRIDO_CONTAINERS_PREFIX}/files/training_cfg.sh"

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit
fi
if [ -z "$WANDB_KEY" ]; then
    echo "WANDB_KEY variable has not been seen. Please set it to you Wandb key."
    exit
fi

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--set_ulim|-ulim]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
set_ulim=false

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
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

# training_script="/root/ibrido_ws/launch_training.sh"
training_script="launch_training.sh"

if $use_sudo; then
    if $set_ulim; then
        sudo singularity exec \
            --env "WANDB_KEY=$WANDB_KEY"\
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
                --comment $COMMENT \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    else
        sudo singularity exec \
            --env "WANDB_KEY=$WANDB_KEY"\
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $COMMENT \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    fi
else
    if $set_ulim; then
        singularity exec \
            --env "WANDB_KEY=$WANDB_KEY"\
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
                --comment $COMMENT \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    else
        singularity exec \
            --env "WANDB_KEY=$WANDB_KEY"\
            --bind $binddirs\
            --no-mount home,cwd \
            --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
                --robot_pkg_name $RB_PNAME \
                --robot_pkg_pref_path $RB_PPREFNAME \
                --num_envs $N_ENVS \
                --ulim_n $ULIM_N \
                --ns $SHM_NS \
                --run_name $RNAME \
                --comment $COMMENT \
                --seed $SEED \
                --timeout_ms $TIMEOUT_MS \
                --codegen_override $CODEGEN_OVERRIDE_BDIR
    fi
fi

