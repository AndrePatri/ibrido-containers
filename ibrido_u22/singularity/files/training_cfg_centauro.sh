#!/bin/bash

export URDF_PATH="\${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="\${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="\${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config.yaml" # path to yaml file for jnt imp configuration
export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export SHM_NS="Centauro" # shared mem namespace used for all shared data on CPU 
# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export N_ENVS=450 # number of env to run in parallel
export RNAME="CentauroLinVelTrack" # a descriptive base name for this run
export SEED=566 # random n generator seed to be used for this run
export SET_ULIM=1 
export ULIM_N=10000 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=10000 # timeout after which each script autokills ([ms])
export CODEGEN_OVERRIDE_BDIR="none"
# export CODEGEN_OVERRIDE_BDIR="\${HOME}/aux_data/KyonRHCLusterClient_kyon_no_wheels/CodeGen/kyon_no_wheelsRhc" # where to load rhc codegenerated functions
# export COMMENT="${PBS_JOBID}__" # any training comment
export COMMENT="centauro, simple stepping" # any training comment
export WANDB_KEY="" # wandb key for logging remote db data
export CLUSTER_DB=1
export LAUNCH_ROSBAG=1
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export REMOTE_STEPPING=1 
export CUSTOM_ARGS_NAMES="control_wheels"
export CUSTOM_ARGS_DTYPE="bool"
export CUSTOM_ARGS_VALS="false"