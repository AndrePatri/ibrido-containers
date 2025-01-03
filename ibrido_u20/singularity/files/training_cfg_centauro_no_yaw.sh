#!/bin/bash
export XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/xbot2_basic.yaml"
# export XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic_wheels.yaml"
export XMJ_FILES_DIR="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files"

# export SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace 
export SHM_NS="centauro_big_wheels" # shared mem namespace used for all shared data on CPU 
export N_ENVS=1 # number of env to run in parallel
export RNAME="LinVelTrackBaseline" # a descriptive base name for this run
export SEED=1 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export ACTOR_LWIDTH=256
export ACTOR_DEPTH=2
export CRITIC_LWIDTH=512
export CRITIC_DEPTH=4
export OBS_NORM=1
export OBS_RESCALING=0
export IS_CLOSED_LOOP=1
export COMMENT='centauro big wheels' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open_with_ub.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_with_ub.yaml"
fi
export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.04
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

# export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback use_diff_vels xmj_timeout xmj_files_dir state_from_xbot closed_partial"
# export CUSTOM_ARGS_DTYPE="bool bool bool bool bool int string bool bool"
# export CUSTOM_ARGS_VALS="false true true false false $TIMEOUT_MS $HOME/ibrido_ws/src/$XMJ_FILES_DIR true true"
# export REMOTE_ENV_FNAME="lrhcontrolenvs.envs.xmj_env"

export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback use_diff_vels state_from_xbot rt_safety_perf_coeff closed_partial add_upper_body"
export CUSTOM_ARGS_DTYPE="bool bool bool bool bool bool float bool bool"
export CUSTOM_ARGS_VALS="false true true false false true 1.0 true true"
export REMOTE_ENV_FNAME="lrhcontrolenvs.envs.rt_deploy_env"

export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_IP="127.0.0.1"

# export ROS_MASTER_URI="http://10.24.4.100:11311" # centauro embeeded
# export ROS_IP=$(hostname -I | awk 'print $1') # hostname 