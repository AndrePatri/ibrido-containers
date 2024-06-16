#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${SCRIPT_DIR}/files/bind_list.sh"

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_BDIRS[*]}"
unset IFS # Reset the internal field separator

singularity exec \
    -B /tmp/.X11-unix:/tmp/.X11-unix\
    -B /etc/localtime:/etc/localtime:ro \
    --bind $binddirs\
    --no-mount home,cwd \
    --nv ibrido_isaac.sif bash