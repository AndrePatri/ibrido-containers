#!/bin/bash
# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="kyon_no_wheels" # shared mem namespace used for all shared data on CPU 
export N_ENVS=600 # number of env to run in parallel
export RNAME="LinVelTrackBaseline" # a descriptive base name for this run
export SEED=566 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export ACTOR_LWIDTH=256
export ACTOR_DEPTH=2
export CRITIC_LWIDTH=512
export CRITIC_DEPTH=4
export OBS_NORM=1
export OBS_RESCALING=0
export COMMENT='kyon no wheels, obs norm, no obs rescale, action repeat x1, after nn init refactor, added back flight info' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-ros-pkg/kyon_srdf/srdf/kyon.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config.yaml" # path to yaml file for jnt imp configuration
export CLUSTER_CL_FNAME="kyonrlstepping.controllers.horizon_based.kyon_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
export LAUNCH_ROSBAG=0
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export SRDF_PATH_ROSBAG="${HOME}/aux_data/KyonRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="wheels fixed_flights deact_when_failure"
export CUSTOM_ARGS_DTYPE="xacro bool bool"
export CUSTOM_ARGS_VALS="false true true"
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=10000 # timeout after which each script autokills ([ms])