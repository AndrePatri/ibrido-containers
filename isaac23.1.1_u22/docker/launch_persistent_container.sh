#!/bin/bash

IMAGE_NAME="lrhc:isaac2023.1.1"
CONTAINER_NAME="lrhc_isaac2023.1.1"

# Check if the container exists
docker container inspect "$CONTAINER_NAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then # if the previous command failed
    # Create the container with GPU support and other options
    docker create --gpus all -it \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=N" \
        --entrypoint bash \
        -v ~/docker/lrhc-docker/isaac-sim/cache/kit:/root/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/cache:rw \
        -v ~/docker/lrhc-docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/lrhc-docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/lrhc-docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/lrhc-docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/lrhc-docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/lrhc-docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/lrhc-docker/isaac-sim/documents:/root/Documents:rw \
        -v ~/docker/lrhc-docker/RL_ws:/root/RL_ws:rw \
        -v ~/docker/lrhc-docker/results:/root/results:rw \
        --name "$CONTAINER_NAME" \
        ${IMAGE_NAME} 
fi

# Start the already-created container
docker start -i "$CONTAINER_NAME"

