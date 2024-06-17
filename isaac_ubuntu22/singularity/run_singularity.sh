#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${SCRIPT_DIR}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s]"
    exit 1
}
use_sudo=false # whether to use superuser privileges

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
    esac
    shift
done

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

if $use_sudo; then
    sudo singularity exec \
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_isaac.sif bash
else
    singularity exec \
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ibrido_isaac.sif bash
fi

