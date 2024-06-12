#!/bin/bash

IBRIDO_PREFIX=$HOME/docker/ibrido-singularity
# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# sudo singularity exec \
#     -B ~/docker/ibrido-docker/aux_data:/root/aux_data:rw \
#     -B ~/docker/ibrido-docker/conda:/opt/conda:rw \
#     -B ~/docker/ibrido-docker/conda_hidden/.conda:/root/.conda:rw \
#     -B ~/docker/ibrido-docker/.cache:/root/.cache:rw \
#     -B ~/docker/ibrido-docker/isaac-cache/isaac-sim/kit:/isaac-sim/kit/cache:rw \
#     -B ~/docker/ibrido-docker/isaac-cache/isaac-sim/kit:/isaac-sim/kit/cache:rw \
#     -B ~/docker/ibrido-docker/isaac-cache/isaac-sim/ov:/root/.cache/ov:rw \
#     -B ~/docker/ibrido-docker/isaac-cache/isaac-sim/glcache:/root/.cache/nvidia/GLCache:rw \
#     -B ~/docker/ibrido-docker/isaac-cache/isaac-sim/computecache:/root/.nv/ComputeCache:rw \
#     -B ~/docker/ibrido-docker/isaac-sim/documents:/root/Documents:rw \
#     -B ~/docker/ibrido-docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
#     -B ~/docker/ibrido-docker/isaac-sim/data:/root/.local/share/ov/data:rw \
#     -B /tmp/.X11-unix:/tmp/.X11-unix \
#     -B /etc/localtime:/etc/localtime:ro \
#     -B ~/docker/ibrido-docker/ibrido_ws:/root/ibrido_ws:rw \
#     -B ~/docker/ibrido-docker/training_data:/root/training_data:rw \
#     --no-home \
#     --nv ibrido_container_isaac.sif bash

sudo singularity exec \
    -B /tmp/.X11-unix:/tmp/.X11-unix\
    -B /etc/localtime:/etc/localtime:ro \
    -B ${IBRIDO_PREFIX}/ibrido_ws:/root/ibrido_ws:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/exts:/isaac-sim/kit/exts/:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/logs:/isaac-sim/kit/logs:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/data:/isaac-sim/kit/data:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/warp:/root/.cache/warp:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/local:/root/.local:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/documents:/root/Documents:rw \
    --no-home \
    --nv ibrido_container_isaac.sif bash

    # -B ${IBRIDO_PREFIX}/isaac-sim/data:/root/.local/share/ov/data:rw \
