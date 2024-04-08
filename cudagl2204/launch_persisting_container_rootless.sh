#!/bin/bash

# Default container name
DEFAULT_CONTAINER_NAME="mycudagl-2204-cudagl-basic"

# Check if an argument is provided for the container name
if [ -z "$1" ]; then
    # If no container name is provided, use the default
    CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
else
    # If a container name is provided, use it
    CONTAINER_NAME="$1"
fi

# Check if the container exists
docker container inspect "$CONTAINER_NAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then # if the previous command failed
    # Create the container with GPU support and other options
    docker create --gpus all -it \
         --mount type=bind,source="$HOME",target=/home/host \
         --name "$CONTAINER_NAME" \
         -p 9422:9422 \
         crizzard/lr-gym:2204-cudagl-basic \
         bash
fi

# Start the already-created container
docker start -i "$CONTAINER_NAME"

