#!/bin/bash
export SHM_NS="Ibrido" # shared mem namespace used for all shared data on CPU 
export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export N_ENVS=1 # number of env to run in parallel
export RNAME="IbridoRun" # a descriptive base name for this run
export SEED=0 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export ACTOR_LWIDTH=2
export ACTOR_DEPTH=1
export CRITIC_LWIDTH=1
export CRITIC_DEPTH=1
export OBS_NORM=0
export OBS_RESCALING=1
export COMMENT="\"${PBS_JOBID}: \"" # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/" # path to yaml file for jnt imp configuration
export CLUSTER_CL_FNAME="lrhc_control.rhc.controllers.lrhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
# export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/***RHCLusterClient_***/CodeGen/***Rhc" # where to load rhc codegenerated functions
export LAUNCH_ROSBAG=1
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export ENV_IDX_BAG_DEMO=1
export ENV_IDX_BAG_EXPL=2export SRDF_PATH_ROSBAG="${HOME}/aux_data/KyonRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="dummy_custom_arg"
export CUSTOM_ARGS_DTYPE="str"
export CUSTOM_ARGS_VALS="empty"
export SET_ULIM=1 
export ULIM_N=10000 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=90000 # timeout after which each script autokills ([ms])