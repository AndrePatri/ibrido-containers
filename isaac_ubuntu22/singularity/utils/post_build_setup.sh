#!/bin/bash
set -e # exiting if any cmd fails

echo "Running post-build steps. It may take a while...."
 
${UTILS_SCRIPTPATH}/create_mamba_env.sh 
${UTILS_SCRIPTPATH}/setup_ws.sh

# Launch a shell session
# /bin/bash