#!/bin/bash
source /root/ibrido_files/training_cfgs/joy_cfg.sh
source /root/ibrido_files/training_cfgs/zmq_cfg.sh
export EVAL=1
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=0
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/training_data/d2025_12_20_h13_m32_s48-CentauroCloopPartialUbNoWheelsPertRecovery_StayingAliveEnv"
export MNAME="d2025_12_20_h13_m32_s48-CentauroCloopPartialUbNoWheelsPertRecovery_StayingAliveEnv_model"

export XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/xbot2_basic.yaml"
# export XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic_wheels.yaml"
export XMJ_FILES_DIR="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files"

export RT_DEPLOY=0


if [[ $RT_DEPLOY -eq 1 ]]; then
  # Set ROS_MASTER_URI and ROS_IP for deployment
  export ROS_MASTER_URI="http://10.24.4.100:11311" # Centauro embedded
  export ROS_IP=$(hostname -I | awk '{print $1}') # Extract first IP address
else
  # Set ROS_MASTER_URI and ROS_IP for local setup
  export ROS_MASTER_URI="http://127.0.0.1:11311"
  export ROS_IP="127.0.0.1"
fi

export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_IP="127.0.0.1"

# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="centauro_big_wheels_ub" # shared mem namespace used for all shared data on CPU 
export N_ENVS=1 # number of env to run in parallel
export RNAME="CentauroCLoopPartialNoWheelsActRepAblation" # a descriptive base name for this run
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
export ACTION_REPEAT=4
export USE_SAC=1
export USE_DUMMY=0
export DISCOUNT_FACTOR=0.998
export USE_PERIOD_RESETS=0
export COMMENT='centauro big wheels CLOOP partial, eval on real robot' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open_with_ub.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_with_ub.yaml"
fi
export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.04
export PHYSICS_DT=0.0005
export N_NODES=25
export CLUSTER_DB=1
export CODEGEN_OVERRIDE_BDIR="none"
export TRAIN_ENV_FNAME="staying_alive_env"
export TRAIN_ENV_CNAME="StayingAliveEnv"
export BAG_SDT=3600.0
export BRIDGE_DT=0.05
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export SRDF_PATH_ROSBAG="${HOME}/aux_data/CentauroRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=30000 # timeout after which each script autokills ([ms])

if [[ $RT_DEPLOY -eq 1 ]]; then
  export CUSTOM_ARGS_NAMES="add_remote_exit_flag step_height control_wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_diff_vels state_from_xbot rt_safety_perf_coeff estimate_v_root add_upper_body use_mpc_pos_for_robot torque_correction xbot2_filter_prof use_jnt_v_feedback"
  export CUSTOM_ARGS_DTYPE="bool float bool bool bool bool bool bool bool float bool bool bool float str bool"
  export CUSTOM_ARGS_VALS="true 0.10 false true true false true false true 1.0 false true true 1.0 fast true"
  export REMOTE_ENV_FNAME="aug_mpc_envs.world_interfaces.rt_deploy_world_interface"
  
  export CUSTOM_ARGS_NAMES+=" is_sim" # we need to tell the interface we are in sim
  export CUSTOM_ARGS_DTYPE+=" bool"
  export CUSTOM_ARGS_VALS+=" true"

else
  export CUSTOM_ARGS_NAMES="step_height render_to_file render_fps control_wheels fixed_flights adaptive_is lin_a_feedback use_diff_vels xmj_timeout xmj_files_dir state_from_xbot closed_partial torque_correction xbot2_filter_prof use_jnt_v_feedback add_upper_body"
  export CUSTOM_ARGS_NAMES+=" generate_stepup_terrain ground_type enable_height_sensor height_sensor_pixels height_sensor_resolution enable_height_vis"
  export CUSTOM_ARGS_DTYPE="float bool float bool bool bool bool bool int string bool bool float str bool bool"
  export CUSTOM_ARGS_DTYPE+=" bool str bool int float bool"  
  export CUSTOM_ARGS_VALS="0.10 false 60.0 false true true false false $TIMEOUT_MS $HOME/ibrido_ws/src/$XMJ_FILES_DIR true true 1.0 fast true true"
  export CUSTOM_ARGS_VALS+=" false stepup_prim true 10 0.16 false"
  export REMOTE_ENV_FNAME="aug_mpc_envs.world_interfaces.xmj_world_interface"  
fi
