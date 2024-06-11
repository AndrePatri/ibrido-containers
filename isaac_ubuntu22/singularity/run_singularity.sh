#!/bin/bash

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/files/.env.base

sudo singularity shell \
    -B /tmp/.X11-unix:/tmp/.X11-unix \
    -B /etc/localtime:/etc/localtime:ro \
    -B ~/docker/ibrido-docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -B ~/docker/ibrido-docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -B ~/docker/ibrido-docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -B ~/docker/ibrido-docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -B ~/docker/ibrido-docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -B ~/docker/ibrido-docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -B ~/docker/ibrido-docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -B ~/docker/ibrido-docker/isaac-sim/documents:/root/Documents:rw \
    -B ~/docker/ibrido-docker/ibrido_ws:/root/ibrido_ws:rw \
    -B ~/docker/ibrido-docker/training_data:/root/training_data:rw \
    -B ~/docker/ibrido-docker/aux_data:/root/aux_data:rw \
    -B ~/docker/ibrido-docker/conda:/opt/conda:rw \
    --oci --nv ibrido_isaac.oci.sif

# singularity run --oci ./ibrido.oci.sif
# singularity shell \
#     -B ~/docker/ibrido-docker/ibrido_ws:/root/ibrido_ws:rw \
#     -B ~/docker/ibrido-docker/training_data:/root/training_data:rw \
#     -B ~/docker/ibrido-docker/aux_data:/root/aux_data:rw \
#     -B ~/docker/ibrido-docker/conda:/opt/conda:rw \
#     --oci --nv ibrido.oci.sif