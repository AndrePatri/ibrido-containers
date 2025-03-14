#!/bin/bash
export EVAL=0
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/training_data/"
export MNAME=""

export WANDB_KEY="25f235316292344cea6dfa68e7c95409b3374d03"
export SHM_NS="kyon_wheels" # shared mem namespace used for all shared data on CPU 
export N_ENVS=800 # number of env to run in parallel
export RNAME="" # a descriptive base name for this run
export SEED=56 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export COMPRESSION_RATIO=0.6
export ACTOR_LWIDTH=128
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256
export CRITIC_DEPTH=3
export OBS_NORM=1
export OBS_RESCALING=0
export CRITIC_ACTION_RESCALE=1
export WEIGHT_NORM=1
export LAYER_NORM=0
export BATCH_NORM=0
export IS_CLOSED_LOOP=1
export DUMP_ENV_CHECKPOINTS=1
export TOT_STEPS=5000000
export USE_RND=0
export DEMO_ENVS_PERC=0.0
export DEMO_STOP_THRESH=10.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=3
export USE_SAC=1
export DISCOUNT_FACTOR=0.994
export USE_PERIOD_RESETS=0
export COMMENT='kyon wheels FULLY CLOSED, UTD 16, action rep 3, fake pos track (action rate, CoT, dir track)' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-ros-pkg/kyon_srdf/srdf/kyon.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config.yaml"
fi

export CLUSTER_CL_FNAME="kyonrlstepping.controllers.horizon_based.kyon_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.03
export N_NODES=31
export CLUSTER_DB=1
export PHYSICS_DT=0.001
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/KyonRHCLusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc"
# export TRAIN_ENV_FNAME="linvel_env_baseline"
# export TRAIN_ENV_CNAME="LinVelTrackBaseline"
export TRAIN_ENV_FNAME="fake_pos_env_baseline"
export TRAIN_ENV_CNAME="FakePosEnvBaseline"
# export TRAIN_ENV_FNAME="fake_pos_env_variable_flights_with_demo"
# export TRAIN_ENV_CNAME="FakePosEnvVariableFlightsWithDemo"
# export TRAIN_ENV_FNAME="fake_pos_env_with_demo"
# export TRAIN_ENV_CNAME="FakePosEnvWithDemo"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="LinVelEnvWithDemo"
# export TRAIN_ENV_FNAME="variable_flights_baseline"
# export TRAIN_ENV_CNAME="VariableFlightsBaseline"
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=-1
export ENV_IDX_BAG_DEMO=-1
export ENV_IDX_BAG_EXPL=-1
export SRDF_PATH_ROSBAG="${HOME}/aux_data/KyonRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_flat_ground estimate_v_root fully_closed use_gpu" 
export CUSTOM_ARGS_DTYPE="xacro bool bool bool bool bool bool bool bool"
export CUSTOM_ARGS_VALS="true true false false false true false true false"
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=120000 # timeout after which each script autokills ([ms])