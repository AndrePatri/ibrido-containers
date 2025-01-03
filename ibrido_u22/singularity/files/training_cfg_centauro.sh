#!/bin/bash
export WANDB_KEY=""
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
export WEIGHT_NORM=1
export IS_CLOSED_LOOP=1
export COMMENT='centauro big wheels' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
    export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config.yaml"
fi

export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.05
export CLUSTER_DB=1
export PHYSICS_DT=0.0005
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/CentauroRHCLusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc"
# export TRAIN_ENV_FNAME="linvel_env_baseline"
# export TRAIN_ENV_CNAME="LinVelTrackBaseline"
export TRAIN_ENV_FNAME="fake_pos_env_baseline"
export TRAIN_ENV_CNAME="FakePosEnvBaseline"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="LinVelEnvWithDemo"
# export TRAIN_ENV_FNAME="variable_flights_baseline"
# export TRAIN_ENV_CNAME="VariableFlightsBaseline"
export DEMO_STOP_THRESH=2.0
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=-1
export ENV_IDX_BAG_DEMO=-1
export ENV_IDX_BAG_EXPL=-1
export SRDF_PATH_ROSBAG="${HOME}/aux_data/CentauroRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="control_wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_flat_ground estimate_v_root add_upper_body"
export CUSTOM_ARGS_DTYPE="bool bool bool bool bool bool bool bool"
export CUSTOM_ARGS_VALS="false true true false true true false false"
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=10000 # timeout after which each script autokills ([ms])

# job_id=$(echo "$PBS_JOBID" | cut -d'.' -f1) # extract the job ID before the first dot
# export SHM_NS+="_$(date '+%Y_%m_%d_%H_%M_%S')_ID${job_id}" # appending unique string to actual shm namespace 
