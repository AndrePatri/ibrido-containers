#!/bin/bash
set -e # exiting if any cmd fails

echo "Creating mamba environment..."
micromamba env create -y -f /root/mamba_env.yml