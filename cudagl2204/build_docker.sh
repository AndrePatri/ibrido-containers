#!/bin/bash

# Navigate to the directory containing the Dockerfile
cd "$(dirname "$0")"

# Check if an argument is provided for the tag
if [ -z "$1" ]; then
    # If no tag is provided, use a default tag
    TAG="mycudagldocker/mycudagl:2204-cudagl-basic"
else
    # If a tag is provided, use it
    TAG="$1"
fi

# Build the Docker image with the specified tag
docker build --tag "$TAG" .