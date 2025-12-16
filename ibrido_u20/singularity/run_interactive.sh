#!/bin/bash

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_20/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--wandb_key|-w <key>]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
wandb_key=""

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
    singularity exec \
        --fakeroot \
        --cleanenv \
        --net \
        --env "DISPLAY=${DISPLAY}"\
        --env XAUTHORITY="$XAUTHORITY" \
        --env "WANDB_KEY=$wandb_key"\
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_xbot.sif bash
else
    singularity exec \
        --cleanenv \
        --env "DISPLAY=${DISPLAY}"\
        --env "WANDB_KEY=$wandb_key"\
        --env XAUTHORITY="$XAUTHORITY" \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_xbot.sif bash
fi

