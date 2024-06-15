#!/bin/bash
echo 'Insert your generated NVIDIA NGC password/token to pull IsaacSim-->'
singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
echo 'Starting build of IBRIDO singularity container-->'
singularity build --fakeroot ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary

