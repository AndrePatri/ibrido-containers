#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# retrieve dirs to be binded
${SCRIPT_DIR}/utils/create_ws.sh

# export MY_LIST="${my_list[*]}"

# echo 'Insert your generated NVIDIA NGC password/token to pull IsaacSim-->'
# singularity registry login --username \$oauthtoken docker://nvcr.io # run with sudo if singularity build --fakeroot
# echo 'Starting build of IBRIDO singularity container-->'
# sudo singularity build --bind ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary
# # singularity build --fakeroot ./ibrido_isaac.sif ./u22_isaac.def # either --fakeroot or sudo are necessary


