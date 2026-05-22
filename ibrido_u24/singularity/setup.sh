#!/bin/bash
set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_u24/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

usage() {
    echo "Usage: $0 [--build|-b] [--use_sudo|-s] [--init|-i] [--do_setup|-stp] [--update_ws|-uws] [--ngc_key|-ngc <key>] [--with_private_gitdirs|-wpg]"
    exit 1
}

build_container=false
use_sudo=false
init=false
do_setup=false
update_ws=false
with_private_gitdirs=false
ngc_key=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--build) build_container=true ;;
        -s|--use_sudo) use_sudo=true ;;
        -i|--init) init=true ;;
        -stp|--do_setup) do_setup=true ;;
        -uws|--update_ws) update_ws=true ;;
        -wpg|--with_private_gitdirs) with_private_gitdirs=true ;;
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
    echo '--> Please make sure to provide your generated NVIDIA NGC password/token using the --ngc_key arg to pull IsaacSim (see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)'
    if $use_sudo; then
        sudo singularity registry login --username \$oauthtoken --password "$ngc_key" docker://nvcr.io
        echo '--> Starting building of IBRIDO singularity container (sudo)...'
        sudo singularity build "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" "$IBRIDO_CONTAINERS_PREFIX/u24_isaac.def"
    else
        singularity registry login --username \$oauthtoken --password "$ngc_key" docker://nvcr.io
        echo '--> Starting building of IBRIDO singularity container (fakeroot)...'
        singularity build --fakeroot "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" "$IBRIDO_CONTAINERS_PREFIX/u24_isaac.def"
    fi
    echo 'container was built.'
fi

if $init; then
    if $with_private_gitdirs; then
        IBRIDO_CLONE_PRIVATE_GITDIRS=1 "${IBRIDO_CONTAINERS_PREFIX}/utils/create_ws.sh"
    else
        "${IBRIDO_CONTAINERS_PREFIX}/utils/create_ws.sh"
    fi
fi

if $update_ws; then
    "${IBRIDO_CONTAINERS_PREFIX}/utils/update_ws_code.sh"
fi

if $do_setup; then
    echo '--> About to run setup steps (from within container)...'
    IFS=','
    binddirs="${IBRIDO_B_ALL[*]}"
    unset IFS
    singularity exec \
        --env "BYOBU_CONFIG_DIR=/root/.byobu" \
        --bind "$binddirs" \
        --no-mount home,cwd \
        --pwd /root \
        --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" post_build_setup.sh
    echo 'Done. You can now either launch the container with run_interactive.sh or start the training with execute.sh'
fi
