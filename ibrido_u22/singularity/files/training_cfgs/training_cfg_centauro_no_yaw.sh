#!/bin/bash
export EVAL=0
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/training_data/"
export MNAME=""

export WANDB_KEY="25f235316292344cea6dfa68e7c95409b3374d03"
export SHM_NS="centauro_big_wheels_no_yaw" # shared mem namespace used for all shared data on CPU 
export N_ENVS=800 # number of env to run in parallel
export RNAME="" # a descriptive base name for this run
export SEED=98 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export COMPRESSION_RATIO=-1.0
export ACTOR_LWIDTH=128
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256
export CRITIC_DEPTH=4
export OBS_NORM=1
export OBS_RESCALING=0
export CRITIC_ACTION_RESCALE=0
export WEIGHT_NORM=1
export IS_CLOSED_LOOP=1
export DUMP_ENV_CHECKPOINTS=0
export DEMO_STOP_THRESH=10.0
export TOT_STEPS=80000000
export USE_RND=0
export DEMO_ENVS_PERC=0.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=1
export COMMENT='centauro big wheels (fixed ankle yaw)' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open_with_ub.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_with_ub.yaml"
fi

export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.05
export N_NODES=31
export CLUSTER_DB=1
export PHYSICS_DT=0.0005
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/CentauroRHCLusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc"
# export TRAIN_ENV_FNAME="linvel_env_baseline"
# export TRAIN_ENV_CNAME="LinVelTrackBaseline"
export TRAIN_ENV_FNAME="fake_pos_env_variable_flights"
export TRAIN_ENV_CNAME="FakePosEnvVariableFlights"
# export TRAIN_ENV_FNAME="fake_pos_env_variable_flights_with_demo"
# export TRAIN_ENV_CNAME="FakePosEnvVariableFlightsWithDemo"
# export TRAIN_ENV_FNAME="fake_pos_env_with_demo"
# export TRAIN_ENV_CNAME="FakePosEnvWithDemo"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="LinVelEnvWithDemo"
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=96
export ENV_IDX_BAG_DEMO=-1
export ENV_IDX_BAG_EXPL=-1
export SRDF_PATH_ROSBAG="${HOME}/aux_data/CentauroRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback closed_partial fix_yaw use_flat_ground estimate_v_root"
export CUSTOM_ARGS_DTYPE="bool bool bool bool bool bool bool bool"
export CUSTOM_ARGS_VALS="true true true false true true true false"
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=120000 # timeout after which each script autokills ([ms])
