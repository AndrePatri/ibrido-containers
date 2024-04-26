#!/bin/bash

# Navigate to the directory containing the Dockerfile
cd "$(dirname "$0")"

TAG="ibrido:isaac2023.1.1"

# Build the Docker image with the specified tag
docker build --tag "$TAG" .
