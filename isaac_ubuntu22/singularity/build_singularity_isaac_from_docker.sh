#!/bin/bash

#sudo singularity registry login --username \$oauthtoken docker://nvcr.io
# build singularity container based on Dockerfile
sudo singularity build --oci ./ibrido_isaac.oci.sif ./DockerfileIsaac4
