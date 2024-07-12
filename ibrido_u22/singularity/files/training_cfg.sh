#!/bin/bash

export RB_PNAME="kyon"
export RB_PPREFNAME="\${HOME}/ibrido_ws/src/iit-kyon-ros-pkg"
export SHM_NS="IBRIDO"
export N_ENVS=128
export RNAME="LinVelTrack"
export SEED=0
export ULIM_N=28672
export TIMEOUT_MS=60000
export CODEGEN_OVERRIDE_BDIR="\${HOME}/aux_data/KyonRHCLusterClient_kyon0/CodeGen/kyon0Rhc"
export COMMENT="${PBS_JOBID}__"
export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace
export WANDB_KEY=""