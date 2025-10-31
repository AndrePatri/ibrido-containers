#!/bin/bash

export RT_DEPLOY=1

export EVAL=1
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=0
export OVERRIDE_AGENT_REFS=1
# export MPATH="/root/training_data/d2025_10_29_h23_m12_s06-KyonRealPartialCloopNoWheels_FakePosEnvBaseline"
# export MNAME="d2025_10_29_h23_m12_s06-KyonRealPartialCloopNoWheels_FakePosEnvBaseline_model"
export MPATH="/root/training_data/d2025_10_30_h12_m44_s13-KyonRealPartialCloopNoWheels_FakePosEnvBaseline"
export MNAME="d2025_10_30_h12_m44_s13-KyonRealPartialCloopNoWheels_FakePosEnvBaseline_model"

export RESUME=0 # resume a previous training using a checkpoint

export XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic_real.yaml"
export XMJ_FILES_DIR="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real"

# Set ROS_MASTER_URI and ROS_IP for deployment
export ROS_MASTER_URI="http://10.24.13.100:11311" # Centauro embedded
export ROS_IP=$(hostname -I | awk '{print $1}') # Extract first IP address

# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="kyon_real_no_wheels" # shared mem namespace used for all shared data on CPU 
export N_ENVS=1 # number of env to run in parallel
export RNAME="KyonRealCloopPartialNoWheels" # a descriptive base name for this run
export SEED=1 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export COMPRESSION_RATIO=0.6
export ACTOR_LWIDTH=128
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256
export CRITIC_DEPTH=3
export OBS_NORM=1
export OBS_RESCALING=0
export WEIGHT_NORM=1
export LAYER_NORM=0
export BATCH_NORM=0
export IS_CLOSED_LOOP=1
export DUMP_ENV_CHECKPOINTS=1
export DEMO_STOP_THRESH=10.0
export TOT_STEPS=10000
export DEMO_ENVS_PERC=0.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=5
export USE_SAC=1
export DISCOUNT_FACTOR=0.99
export USE_PERIOD_RESETS=0
export COMMENT='kyon REAL no wheels CLOSED partial, real deployment' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-description/kyon_urdf/urdf/kyon.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-kyon-description/kyon_srdf/srdf/kyon.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config_kyon_real_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config_kyon_real.yaml"
fi
export CLUSTER_CL_FNAME="kyonrlstepping.controllers.horizon_based.kyon_real_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.035
export PHYSICS_DT=0.0005
export N_NODES=31
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
# export TRAIN_ENV_FNAME="linvel_env_baseline"
# export TRAIN_ENV_CNAME="LinVelTrackBaseline"
export TRAIN_ENV_FNAME="fake_pos_env_baseline"
export TRAIN_ENV_CNAME="FakePosEnvBaseline"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="LinVelEnvWithDemo"
export BAG_SDT=120.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export SRDF_PATH_ROSBAG="${HOME}/aux_data/KyonRHCLusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=30000 # timeout after which each script autokills ([ms])

export CUSTOM_ARGS_NAMES="step_height wheels fixed_flights adaptive_is lin_a_feedback closed_partial fully_closed estimate_v_root use_jnt_v_feedback base_linkname use_diff_vels xmj_timeout xmj_files_dir state_from_xbot rt_safety_perf_coeff use_mpc_pos_for_robot torque_correction" 
export CUSTOM_ARGS_DTYPE="float xacro bool bool bool bool bool bool bool str bool int string bool float bool float"
export CUSTOM_ARGS_VALS="0.18 false true true false true false false true pelvis false $TIMEOUT_MS $HOME/ibrido_ws/src/$XMJ_FILES_DIR true 1.0 true 1.35" 
export REMOTE_ENV_FNAME="aug_mpc_envs.envs.rt_deploy_env" 