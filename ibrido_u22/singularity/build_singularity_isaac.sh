#!/bin/bash
set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--build|-b] [--use_sudo|-s] [--init|-i] [--do_setup|-stp] [--ngc_key|-ngc <key>]"
    exit 1
}
build_container=false
use_sudo=false # whether to use superuser privileges
init=false # whether to initialize/create the workspace
do_setup=false # whether to perform the post-build setup steps
ngc_key=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--build) build_container=true ;;
        -s|--use_sudo) use_sudo=true ;;
        -i|--init) init=true ;;
        -stp|--do_setup) do_setup=true ;;
        -ngc|--ngc_key) 
            if [[ -n "$2" && "$2" != "-"* ]]; then
                ngc_key=$2
                shift
            else
                echo "Error: --ngc_key requires a non-empty argument."
                usage
            fi
            ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if $build_container; then
    echo '--> Building IBRIDO container...'
    echo '--> Insert your generated NVIDIA NGC password/token to pull IsaacSim (see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)'
    if $use_sudo; then
        sudo singularity registry login --username \$oauthtoken --password $ngc_key docker://nvcr.io  # run with sudo if singularity build --fakeroot
        echo '--> Starting building of IBRIDO singularity container (sudo)...'
        sudo singularity build $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $IBRIDO_CONTAINERS_PREFIX/u22_isaac.def # either --fakeroot or sudo are necessary
    else
        singularity registry login --username \$oauthtoken --password $ngc_key docker://nvcr.io  # run with sudo if singularity build --fakeroot
        echo '--> Starting building of IBRIDO singularity container (fakeroot)...'
        singularity build --fakeroot $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $IBRIDO_CONTAINERS_PREFIX/u22_isaac.def # either --fakeroot or sudo are necessary

    fi
    echo 'Done.'
fi

# ws initialization
if $init; then
    echo '--> Initializing workspace...'
    ${IBRIDO_CONTAINERS_PREFIX}/utils/create_ws.sh
    echo 'Done.'
fi

# ws setup
if $do_setup; then
    echo '--> Running setup steps ...'
    # convert bind dirs to comma-separated list
    IFS=',' # Set the internal field separator to a comma
    binddirs="${IBRIDO_B_ALL[*]}"
    unset IFS # Reset the internal field separator
    singularity exec \
        -B /tmp/.X11-unix:/tmp/.X11-unix\
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif post_build_setup.sh
    echo 'Done. You can now either launch the container with run_interactive.sh or start the training with execute.sh'
fi


