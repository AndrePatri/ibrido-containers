#!/bin/bash
set -e # exiting if any cmd fails

echo "Running post-build steps. It may take a while...."
 
create_mamba_env.sh 
setup_ws.sh

# Byobu Fix for launching BASH instead of SH
echo "Fixing Byobu to launch BASH"
echo 'set -g default-shell /bin/bash' >> ${HOME}/.byobu/.tmux.conf
echo 'set -g default-command /bin/bash' >> ${HOME}/.byobu/.tmux.conf

echo 'Warming up IsaacSim ...'
warmup_isaac.sh