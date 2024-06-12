#!/bin/bash
set -e # exiting if any cmd fails

echo "Creating mamba environment..."

source /usr/local/bin/_activate_current_env.sh # enable mamba for this shell

mamba env create -y --log-level off -f /root/mamba_env.yml