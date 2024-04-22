#!/bin/bash
set -e # exiting if any cmd fails

echo "Running post-build steps. It may take a while...."
 
./create_mamba_env.sh 
./setup_ws.sh

# Launch a shell session
# /bin/bash