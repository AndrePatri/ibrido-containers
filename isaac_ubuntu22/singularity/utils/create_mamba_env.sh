#!/bin/bash
set -e # exiting if any cmd fails

echo "Creating mamba environment..."

micromamba shell init --shell bash --root-prefix=/opt/conda/micromamba

micromamba env create -y -f /root/mamba_env.yml