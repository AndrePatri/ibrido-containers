#!/bin/bash
set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_20/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--build|-b] [--use_sudo|-s] [--init|-i] [--do_setup|-stp] [--update_ws|-uws]"
    exit 1
}
build_container=false
use_sudo=false # whether to use superuser privileges
init=false # whether to initialize/create the workspace
do_setup=false # whether to perform the post-build setup steps
update_ws=false
ngc_key=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--build) build_container=true ;;
        -s|--use_sudo) use_sudo=true ;;
        -i|--init) init=true ;;
        -stp|--do_setup) do_setup=true ;;
        -uws|--update_ws) update_ws=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if $build_container; then
    echo '--> Building IBRIDO container...'
    if $use_sudo; then
        echo '--> Starting building of IBRIDO singularity container (sudo)...'
        sudo singularity build $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif $IBRIDO_CONTAINERS_PREFIX/u20_xbot.def # either --fakeroot or sudo are necessary
    else
        echo '--> Starting building of IBRIDO singularity container (fakeroot)...'
        singularity build --fakeroot $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif $IBRIDO_CONTAINERS_PREFIX/u20_xbot.def # either --fakeroot or sudo are necessary
    fi
    echo 'Done.'
fi

# ws initialization
if $init; then
    echo '--> Initializing workspace...'
    ${IBRIDO_CONTAINERS_PREFIX}/utils/create_ws.sh
    echo 'Done.'
fi

# ws update
if $update_ws; then
    ${IBRIDO_CONTAINERS_PREFIX}/utils/update_ws_code.sh
fi

# ws setup
if $do_setup; then
    echo '--> Running setup steps ...'
    # convert bind dirs to comma-separated list
    IFS=',' # Set the internal field separator to a comma
    binddirs="${IBRIDO_B_ALL[*]}"
    unset IFS # Reset the internal field separator
    # singularity exec \
    #     -B /tmp/.X11-unix:/tmp/.X11-unix\
    #     -B /etc/localtime:/etc/localtime:ro \
    #     --bind $binddirs\
    #     --no-mount home,cwd \
    #     --nv --nvccli $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif post_build_setup.sh
    singularity exec \
        -B /etc/localtime:/etc/localtime:ro \
        --bind $binddirs\
        --no-mount home,cwd \
        --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_xbot.sif post_build_setup.sh

    echo 'Done. You can now either launch the container with run_interactive_xbot.sh or start the training with execute_xbot.sh'
fi
