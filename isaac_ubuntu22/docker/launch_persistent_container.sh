#!/bin/bash

IMAGE_NAME="ibrido:isaac2023.1.1"
CONTAINER_NAME="ibrido_isaac2023.1.1"

# Check if the container exists
docker container inspect "$CONTAINER_NAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then # if the previous command failed
    # Create the container with GPU support and other options
    docker create --gpus all -it \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=N" \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /etc/localtime:/etc/localtime:ro \
        --shm-size=1gb\
        --entrypoint bash \
        -v ~/docker/ibrido-docker/isaac-sim/cache/kit:/root/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/cache:rw \
        -v ~/docker/ibrido-docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/ibrido-docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/ibrido-docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/ibrido-docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/ibrido-docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/ibrido-docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/ibrido-docker/isaac-sim/documents:/root/Documents:rw \
        -v ~/docker/ibrido-docker/ibrido_ws:/root/ibrido_ws:rw \
        -v ~/docker/ibrido-docker/training_data:/root/training_data:rw \
        -v ~/docker/ibrido-docker/aux_data:/root/aux_data:rw \
        -v ~/docker/ibrido-docker/conda:/opt/conda:rw \
        --name "$CONTAINER_NAME" \
        ${IMAGE_NAME} 
fi

# Start the already-created container
docker start -i "$CONTAINER_NAME"

