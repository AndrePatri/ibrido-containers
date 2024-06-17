#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${SCRIPT_DIR}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--init|-i] [--do_setup|-stp] [--wandb_key|-w <key>]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
init=false # whether to initialize/create the workspace
do_setup=true # whether to perform the post-build setup steps
wandb_key=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -i|--init) init=true ;;
        -stp|--do_setup) do_setup=true ;;
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

echo '--> Building IBRIDO singularity container ...'
echo '--> Insert your generated NVIDIA NGC password/token to pull IsaacSim (see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)'
if $use_sudo; then
    sudo singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
    echo '--> Starting building of IBRIDO singularity container (sudo)...'
    sudo singularity build ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary
else
    singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
    echo '--> Starting building of IBRIDO singularity container (fakeroot)...'
    singularity build --fakeroot ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary

fi
echo 'Done.'
echo '--> Running post-build setup steps ...'
if $do_setup; then
    # convert bind dirs to comma-separated list
    IFS=',' # Set the internal field separator to a comma
    binddirs="${IBRIDO_B_ALL[*]}"
    unset IFS # Reset the internal field separator
    singularity exec \
        --env "WANDB_KEY=$wandb_key"\
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv ./ibrido_isaac.sif post_build_setup.sh
fi

echo 'Done. You can now either launch the container with run_interactive.sh or start the training with execute.sh'


