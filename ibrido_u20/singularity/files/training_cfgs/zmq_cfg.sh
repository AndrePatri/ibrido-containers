#!/bin/bash
source /root/ibrido_files/training_cfgs/joy_cfg.sh
# Shared defaults for ZMQ bridge launcher integration.
export LAUNCH_ZMQ_BRIDGE=0
export ZMQ_BRIDGE_CORES="-1" # "-1" or -1 disables --cores passing
export ZMQ_BRIDGE_DT=0
export ZMQ_BRIDGE_RHC_INTERNAL=0 # add internal MPC data
export ZMQ_BRIDGE_FULL_DATA=0 # if 1 add training data
