#!/bin/bash

sudo singularity registry login --username \$oauthtoken docker://nvcr.io

# build singularity container based on Dockerfile
singularity build --oci ./ibrido.oci.sif ./Dockerfile