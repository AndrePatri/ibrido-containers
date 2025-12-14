#!/bin/bash
export EVAL=0
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=0
export OVERRIDE_AGENT_REFS=
export MPATH="/root/training_data/"
export MNAME=""

export WANDB_KEY="25f235316292344cea6dfa68e7c95409b3374d03"
export SHM_NS="centauro_big_wheels_ub" # shared mem namespace used for all shared data on CPU 
export N_ENVS=800 # number of env to run in parallel
export RNAME="CentauroCloopPartialUbNoWheelsPercep" # a descriptive base name for this run
export SEED=843 # random n generator seed to be used for this run
export REMOTE_STEPPING=1
export COMPRESSION_RATIO=0.6
export ACTOR_LWIDTH=128
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256
export CRITIC_DEPTH=4
export OBS_NORM=1
export OBS_RESCALING=0
export CRITIC_ACTION_RESCALE=1
export WEIGHT_NORM=1
export LAYER_NORM=0
export BATCH_NORM=0
export IS_CLOSED_LOOP=1
export DEBUG=1
export RMDEBUG=1
export DUMP_ENV_CHECKPOINTS=1
export DEMO_STOP_THRESH=10.0
export TOT_STEPS=15000000
export USE_RND=0
export DEMO_ENVS_PERC=0.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=4
export USE_SAC=1
export USE_DUMMY=0
export DISCOUNT_FACTOR=0.98
export USE_PERIOD_RESETS=0
export COMMENT='centauro big wheels CLOSED with upper body, clipping rewards lb, separate discrete/cont targets (-0.1, -2.0), actions bounds refactor, flight info fix/addition, no rand pert, 10cm steps, smaller platforms, steup with coll prims, heightmap obs, FULL flight control, no omega tracking, UTD 8, track/CoT (x3 scale)/a rate, logstd log(5), 0.25 max step height' # any training comment
export URDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/iit-centauro-ros-pkg/centauro_srdf/srdf/centauro.srdf.xacro" # base path where the description package for the robot are located
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then
    export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/CentauroHybridMPC/centaurohybridmpc/config/jnt_imp_config.yaml"
fi

export CLUSTER_CL_FNAME="centaurohybridmpc.controllers.horizon_based.centauro_rhc_cluster_client" # base path where the description package for the robot are located
export CLUSTER_DT=0.04
export N_NODES=25
export CLUSTER_DB=1
export PHYSICS_DT=0.0005
export USE_GPU_SIM=1
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/CentauroRHCLusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc"
export TRAIN_ENV_FNAME="fake_pos_track_env_phase_control"
export TRAIN_ENV_CNAME="FakePosTrackEnvPhaseControl"
export PUB_HEIGHTMAP=1
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=5
export ENV_IDX_BAG_DEMO=-1
export ENV_IDX_BAG_EXPL=-1
export SRDF_PATH_ROSBAG="${HOME}/aux_data/CentauroRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located
export CUSTOM_ARGS_NAMES="rendering_dt use_random_pertub use_jnt_v_feedback step_height control_wheels fixed_flights adaptive_is \
lin_a_feedback closed_partial fix_yaw use_flat_ground estimate_v_root self_collide add_upper_body \
ground_type enable_height_sensor height_sensor_pixels height_sensor_resolution enable_height_vis"
export CUSTOM_ARGS_DTYPE="float bool bool float bool bool bool bool bool bool bool bool bool bool str bool int float bool"
export CUSTOM_ARGS_VALS="0.1 false false 0.1 false true true false true true false false false true stepup_prim true 10 0.16 false"
export SET_ULIM=1 
export ULIM_N=28672 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=120000 # timeout after which each script autokills ([ms])
