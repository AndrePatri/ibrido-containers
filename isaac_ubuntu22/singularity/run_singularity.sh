#!/bin/bash

IBRIDO_PREFIX=$HOME/docker/ibrido-singularity
# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

singularity exec \
    -B /tmp/.X11-unix:/tmp/.X11-unix\
    -B /etc/localtime:/etc/localtime:ro \
    -B ${IBRIDO_PREFIX}/aux_data:/root/aux_data:rw \
    -B ${IBRIDO_PREFIX}/training_data:/root/training_data:rw \
    -B ${IBRIDO_PREFIX}/ibrido_ws:/root/ibrido_ws:rw \
    -B ${IBRIDO_PREFIX}/conda:/opt/conda:rw \
    -B ${IBRIDO_PREFIX}/conda_hidden/.conda:/root/.conda:rw \
    -B ${IBRIDO_PREFIX}/.cache/wandb:/root/.cache/wandb:rw \
    -B ${IBRIDO_PREFIX}/network/.netrc:/root/.netrc:rw \
    -B ${IBRIDO_PREFIX}/.byobu:/root/.byobu:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/warp:/root/.cache/warp:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/local:/root/.local:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/documents:/root/Documents:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/nv_shadercache:/isaac-sim/kit/exts/omni.gpu_foundation/cache/nv_shadercache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/data:/isaac-sim/kit/data:rw \
    -B ${IBRIDO_PREFIX}/isaac-sim/kitlogs/Isaac-Sim:/isaac-sim/kit/logs/Kit/Isaac-Sim:rw \
    --no-mount home,cwd \
    --nv ibrido_isaac.sif bash