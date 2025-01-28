#!/bin/bash
export EVAL=1
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/training_data/SAC/FakePosEnvWithDemo/d2025_01_27_h13_m17_s49-FakePosEnvWithDemo"
export MNAME="d2025_01_27_h13_m17_s49-FakePosEnvWithDemo_model_checkpoint2386"

export XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/xbot2_basic.yaml"
# export XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic_wheels.yaml"
export XMJ_FILES_DIR="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files"

export RT_DEPLOY=1

if [[ $RT_DEPLOY -eq 1 ]]; then
  # Set ROS_MASTER_URI and ROS_IP for deployment
  export ROS_MASTER_URI="http://10.24.4.100:11311" # Centauro embedded
  export ROS_IP=$(hostname -I | awk '{print $1}') # Extract first IP address
else
  # Set ROS_MASTER_URI and ROS_IP for local setup
  export ROS_MASTER_URI="http://127.0.0.1:11311"
  export ROS_IP="127.0.0.1"
fi

# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="centauro_big_wheels_open" # shared mem namespace used for all shared data on CPU 
export N_ENVS=1 # number of env to run in parallel
export RNAME="LinVelTrackBaseline" # a descriptive base name for this run
export SEED=1 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export COMPRESSION_RATIO=0.8
export ACTOR_LWIDTH=128
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256
export CRITIC_DEPTH=4
export OBS_NORM=1
export OBS_RESCALING=0
export WEIGHT_NORM=1
export IS_CLOSED_LOOP=0
export DUMP_ENV_CHECKPOINTS=0
export DEMO_STOP_THRESH=10.0
export TOT_STEPS=1000000
export DEMO_ENVS_PERC=0.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=3
export COMMENT='centauro big wheels' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config.yaml"
fi
export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.05
export N_NODES=31
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
# export TRAIN_ENV_FNAME="linvel_env_baseline"
# export TRAIN_ENV_CNAME="LinVelTrackBaseline"
export TRAIN_ENV_FNAME="fake_pos_env_baseline"
export TRAIN_ENV_CNAME="FakePosEnvBaseline"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="LinVelEnvWithDemo"
export BAG_SDT=3600.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export SRDF_PATH_ROSBAG="${HOME}/aux_data/CentauroRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=30000 # timeout after which each script autokills ([ms])

if [[ $RT_DEPLOY -eq 1 ]]; then
  export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_diff_vels state_from_xbot rt_safety_perf_coeff estimate_v_root add_upper_body use_mpc_pos_for_robot"
  export CUSTOM_ARGS_DTYPE="bool bool bool bool bool bool bool float bool bool bool"
  export CUSTOM_ARGS_VALS="false true true false true false true 1.0 false false true"
  export REMOTE_ENV_FNAME="lrhcontrolenvs.envs.rt_deploy_env"
else
  export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback use_diff_vels xmj_timeout xmj_files_dir state_from_xbot closed_partial"
  export CUSTOM_ARGS_DTYPE="bool bool bool bool bool int string bool bool"
  export CUSTOM_ARGS_VALS="false true true false false $TIMEOUT_MS $HOME/ibrido_ws/src/$XMJ_FILES_DIR true true"
  export REMOTE_ENV_FNAME="lrhcontrolenvs.envs.xmj_env"  
fi

