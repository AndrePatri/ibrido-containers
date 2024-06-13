#!/bin/bash
echo 'Insert your generated NVIDIA NGC password/token to pull IsaacSim-->'
sudo singularity registry login --username \$oauthtoken docker://nvcr.io
echo 'Starting build of IBRIDO singularity container-->'
sudo singularity build ./ibrido_container_isaac.sif ./u22_isaac.def

