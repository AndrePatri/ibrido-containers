#!/bin/bash

singularity registry login --username myuser nvcr.io
# build singularity container based on Dockerfile
singularity build --oci --keep-layers ./ibrido.oci.sif ./Dockerfile