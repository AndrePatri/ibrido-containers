#!/bin/bash
export EVAL=0
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1
export MPATH="/root/training_data/"
export MNAME=""

export RESUME=0 # resume a previous training using a checkpoint

export WANDB_KEY="${WANDB_KEY:-}"
export SHM_NS="talos" # shared mem namespace used for all shared data on CPU
export N_ENVS=1 # number of envs to run in parallel
export N_CONTACTS=2 # two high-level MPC/shared-memory contacts: left and right sole
export RNAME="TalosOpenDebug" # a descriptive base name for this run
export SEED=7383 # random number generator seed to be used for this run
export REMOTE_STEPPING=0
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
export DEBUG=1
export RMDEBUG=1
export DUMP_ENV_CHECKPOINTS=1
export TOT_STEPS=1000000
export USE_RND=0
export DEMO_ENVS_PERC=0.0
export DEMO_STOP_THRESH=10.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=4
export USE_SAC=1
export USE_DUMMY=0
export DISCOUNT_FACTOR=0.99
export USE_PERIOD_RESETS=0
export COMMENT='talos scaffold, one env, no remote stepping'

export URDF_PATH="${HOME}/ibrido_ws/src/talos-description/talos_description/robots/talos_full_v2.urdf.xacro"
export SRDF_PATH="${HOME}/ibrido_ws/src/talos-description/talos_description/srdf/talos.srdf"

export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/TalosHybridMPC/taloshybridmpc/config/jnt_imp_config_open.yaml"
if (( $IS_CLOSED_LOOP )); then
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/TalosHybridMPC/taloshybridmpc/config/jnt_imp_config.yaml"
fi

export CLUSTER_CL_FNAME="taloshybridmpc.controllers.horizon_based.talos_rhc_cluster_client"
export CLUSTER_DT=0.03
export N_NODES=31
export CLUSTER_DB=1
export PHYSICS_DT=0.001
export USE_GPU_SIM=1
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/TalosRHCLusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc"

# export TRAIN_ENV_FNAME="twist_tracking_env"
# export TRAIN_ENV_CNAME="TwistTrackingEnv"
export TRAIN_ENV_FNAME="fake_pos_tracking_env"
export TRAIN_ENV_CNAME="FakePosTrackingEnv"

export PUB_HEIGHTMAP=0
export BAG_SDT=90.0
export BRIDGE_DT=0.1
export DUMP_DT=50.0
export ENV_IDX_BAG=0
export ENV_IDX_BAG_DEMO=-1
export ENV_IDX_BAG_EXPL=-1
export SRDF_PATH_ROSBAG="${HOME}/aux_data/TalosRHCLusterClient_${SHM_NS}/$SHM_NS.srdf"

# Match the effective Centauro closed-loop mode: partial closed loop. Keep
# adaptive_is disabled explicitly to avoid relying on HybridQuadRhc precedence.
export CUSTOM_ARGS_NAMES="foot_collision head_type flexibility use_fixed_base use_sim enable_crane disable_gazebo_camera use_capsule_collision multiple gazebo_version step_height add_upper_body initial_force_load_divisor adaptive_is closed_partial fully_closed estimate_v_root use_jnt_v_feedback spawning_height use_flat_ground ground_type self_collide"
export CUSTOM_ARGS_DTYPE="xacro xacro xacro xacro xacro xacro xacro xacro xacro xacro float bool float bool bool bool bool bool float bool str bool"
export CUSTOM_ARGS_VALS="thinbox default False false true false true false false classic 0.08 true 2.0 false true false false false 0.95 true flat false"

export SET_ULIM=1
export ULIM_N=131072 # maximum number of open file descriptors for each process (shared memory)
export TIMEOUT_MS=300000 # timeout after which each script autokills ([ms])
