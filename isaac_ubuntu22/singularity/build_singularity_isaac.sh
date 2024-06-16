#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${SCRIPT_DIR}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--init|-i] [--wandb_key|-w <key>]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
init=false # whether to initialize/create the workspace
wandb_key=""

# Parse command line options
# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -i|--init) init=true ;;
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

if $init; then
    ${SCRIPT_DIR}/utils/create_ws.sh
fi

echo '--> Insert your generated NVIDIA NGC password/token to pull IsaacSim (see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)'
if $use_sudo; then
    sudo singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
    echo '--> Starting building of IBRIDO singularity container (sudo)...'
    sudo singularity build --build-arg wandb_key=${wandb_key} ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary
else
    singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
    echo '--> Starting building of IBRIDO singularity container (fakeroot)...'
    singularity build --fakeroot --build-arg wandb_key=${wandb_key} ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary

fi


