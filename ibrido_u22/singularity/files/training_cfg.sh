#!/bin/bash

export RB_PNAME=""
export RB_PPREFNAME="\${HOME}/ibrido_ws/src/"
export SHM_NS="IBRIDO"
export N_ENVS=1
export RNAME="LinVelTrack"
export SEED=0
export ULIM_N=28672
export TIMEOUT_MS=60000
export CODEGEN_OVERRIDE_BDIR=""
export COMMENT="${PBS_JOBID}__"
export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace
export WANDB_KEY="25f235316292344cea6dfa68e7c95409b3374d03"