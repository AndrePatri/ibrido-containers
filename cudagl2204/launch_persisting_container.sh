#!/bin/bash

# Default image name
DEFAULT_IMAGE_NAME="mycudagldocker/mycudagl:2204-cudagl-basic"

# Default container name
DEFAULT_CONTAINER_NAME="mycudagl-2204-cudagl-basic"

# Check if an argument is provided for the image name
if [ -z "$1" ]; then
    # If no image name is provided, use the default
    IMAGE_NAME="$DEFAULT_IMAGE_NAME"
else
    # If an image name is provided, use it
    IMAGE_NAME="$1"
fi

# Check if an argument is provided for the container name
if [ -z "$2" ]; then
    # If no container name is provided, use the default
    CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
else
    # If a container name is provided, use it
    CONTAINER_NAME="$2"
fi

# Check if the container exists
docker container inspect "$CONTAINER_NAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then # if the previous command failed
    # Create the container with GPU support and other options
    docker create --gpus all -it \
         --net=host \
         --mount type=bind,source="$HOME",target=/home/host \
         --name "$CONTAINER_NAME" \
         "$IMAGE_NAME" \
         bash
fi

# Start the already-created container
docker start -i "$CONTAINER_NAME"

