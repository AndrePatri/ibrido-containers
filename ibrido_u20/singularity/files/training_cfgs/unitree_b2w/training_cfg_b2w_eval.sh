#!/bin/bash
export EVAL=1
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/IBRIDO/B2WPartialCloopNoWheelsActRepAblation/d2025_03_22_h09_m18_s23-B2WPartialCloopNoWheelsActRepAblation_FakePosTrackingEnv"
export MNAME="d2025_03_22_h09_m18_s23-B2WPartialCloopNoWheelsActRepAblation_FakePosTrackingEnv_model"

export XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w/xbot2_basic.yaml"
export XMJ_FILES_DIR="KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w"

export RT_DEPLOY=1

# Set ROS_MASTER_URI and ROS_IP for local setup
export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_IP="127.0.0.1"

# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="unitree_b2w" # shared mem namespace used for all shared data on CPU 
export N_ENVS=1 # number of env to run in parallel
export RNAME="B2WPartialCloopNoWheelsActRepAblation" # a descriptive base name for this run
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
export USE_DUMMY=0
export DISCOUNT_FACTOR=0.998
export USE_PERIOD_RESETS=0
export COMMENT='unitree b2w, CLOSED partial, transfer eval on xbot_mujoco' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/unitree_ros/robots/b2w_description/xacro/robot.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/unitree_ros/robots/b2w_description/xacro/robot.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config_open_b2w.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/KyonRLStepping/kyonrlstepping/config/jnt_imp_config_b2w.yaml"
fi
export CLUSTER_CL_FNAME="kyonrlstepping.controllers.horizon_based.b2w_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.03
export PHYSICS_DT=0.0002
export N_NODES=31
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
# export TRAIN_ENV_FNAME="twist_tracking_env"
# export TRAIN_ENV_CNAME="TwistTrackingEnv"
export TRAIN_ENV_FNAME="fake_pos_tracking_env"
export TRAIN_ENV_CNAME="FakePosTrackingEnv"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="TwistTrackingEnvWithDemo"
export BAG_SDT=120.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export SRDF_PATH_ROSBAG="${HOME}/aux_data/B2WRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=30000 # timeout after which each script autokills ([ms])

export CUSTOM_ARGS_NAMES="render_to_file render_fps use_diff_vels xmj_timeout xmj_files_dir state_from_xbot wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_flat_ground estimate_v_root fully_closed control_wheels base_link_name"
export CUSTOM_ARGS_DTYPE="bool float bool int string bool xacro bool bool bool bool bool bool bool bool str"
export CUSTOM_ARGS_VALS="false 60.0 false $TIMEOUT_MS $HOME/ibrido_ws/src/$XMJ_FILES_DIR false true true true false true true false false true base"
export REMOTE_ENV_FNAME="aug_mpc_envs.world_interfaces.xmj_world_interface"  

