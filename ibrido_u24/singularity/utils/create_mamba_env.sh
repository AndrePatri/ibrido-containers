#!/bin/bash
set -e # exiting if any cmd fails

echo "Activating micromamba and creating ${MAMBA_ENV_NAME} environment..."

source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell

micromamba env create -y --log-level error -f ${MAMBA_ENV_FPATH}

echo "Activating micromamba and creating auxiliary ${MAMBA_ENV_FPATH_ISAAC} environment..."

micromamba env create -y --log-level error -f ${MAMBA_ENV_FPATH_ISAAC}