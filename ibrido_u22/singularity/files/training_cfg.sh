#!/bin/bash

export RB_PNAME="kyon" # name of the description package for the robot
export RB_PPREFNAME="\${HOME}/ibrido_ws/src/******" # base path where the description package for the robot are located
export COCLUSTER_DIR="\${HOME}/ibrido_ws/src/******" # base dir where launch_control_cluster.py script is located
export SHM_NS="IBRIDO" # shared mem namespace used for all shared data on CPU 
export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export N_ENVS=128 # number of env to run in parallel
export RNAME="IBRIDO" # a descriptive base name for this run
export SEED=0 # random n generator seed to be used for this run
export ULIM_N=60000 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=300000 # timeout after which each script autokills ([ms])
export CODEGEN_OVERRIDE_BDIR="\${HOME}/aux_data/***RHCLusterClient_***/CodeGen/***Rhc" # where to load rhc codegenerated functions
export COMMENT="${PBS_JOBID}__" # any training comment
export WANDB_KEY="" # wandb key for logging remote db data