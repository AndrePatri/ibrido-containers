#!/bin/bash
set -e # exiting if any cmd fails

echo "Activating mamba and creating ${MAMBA_ENV_NAME} environment..."

source ${MAMBA_EXE_PREFIX}/_activate_current_env.sh # enable mamba for this shell

micromamba env create -y --log-level error -f ${MAMBA_ENV_FPATH}