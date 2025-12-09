#!/bin/bash
export EVAL=0
export DET_EVAL=1
export EVAL_ON_CPU=1
export OVERRIDE_ENV=1
export OVERRIDE_AGENT_REFS=1

export RESUME=0 # resume a previous training using a checkpoint

export MPATH="/root/training_data/"
export MNAME=""

export WANDB_KEY="" # add your wandb key here
export SHM_NS="my_robot" # shared mem namespace used for all shared data on CPU 
export N_ENVS=800 # number of env to run in parallel
export RNAME="MyTrainingRun" # a descriptive base name for this run
export SEED=34 # random n generator seed to be used for this run
export REMOTE_STEPPING=1 # if 0 do not run remote training env (just sim + MPC cluster), useful for MPC dev and db
export COMPRESSION_RATIO=0.6 # agent layer width will be this % smaller than the input size, to encourage information compression
export ACTOR_LWIDTH=128 # only used if COMPRESSION_RATIO=-1
export ACTOR_DEPTH=3
export CRITIC_LWIDTH=256 # only used if COMPRESSION_RATIO=-1
export CRITIC_DEPTH=4
export OBS_NORM=1 # whether to normalize observations
export OBS_RESCALING=0
export CRITIC_ACTION_RESCALE=1
export WEIGHT_NORM=1 # apply agent weight normalization
export LAYER_NORM=0
export BATCH_NORM=0
export IS_CLOSED_LOOP=1 # run mpc cluster in closed loop
export DUMP_ENV_CHECKPOINTS=1 # dump checkpoints from training env
export TOT_STEPS=14000000 # total steps to be collected during training
export USE_RND=0
export DEMO_ENVS_PERC=0.0
export DEMO_STOP_THRESH=10.0
export EXPL_ENVS_PERC=0.0
export ACTION_REPEAT=5 # agent freq wrt cluster freq (1 -> same rate MPC, 2 -> 1 agent action every 2 MPC solutions)
export USE_SAC=1 # use SAC, otherwise PPO
export USE_DUMMY=0 # if set, overrides SAC/PPO and uses dummy agent
export DISCOUNT_FACTOR=0.99
export USE_PERIOD_RESETS=0 # reset periodically agent (can be useful to prevent overfitting to early experience due to net plasticity)
export COMMENT='' # any comment to describe the training run
export URDF_PATH="${HOME}/ibrido_ws/src/myrobot-description/robot_urdf/urdf/my_robot.urdf.xacro" # name of the description package for the robot
export SRDF_PATH="${HOME}/ibrido_ws/src/myrobot-description/robot_urdf/urdf/my_robot.srdf.xacro" # base path where the semantic description package for the robot are located (used for homing)
export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/MyRobotCfg/config/jnt_imp_config_open.yaml" # path to yaml file for jnt imp configuration
if (( $IS_CLOSED_LOOP )); then # jnt imp cfg for closed loop operation
  export JNT_IMP_CF_PATH="${HOME}/ibrido_ws/src/MyRobotCfg/config/jnt_imp_config_cloop.yaml"
fi

export CLUSTER_CL_FNAME="myrobotcontrol.controllers.horizon_based.my_robot_cluster_client" # file name where cluster client implementation is located (package should be installed)
export CLUSTER_DT=0.035 # rate at which MPCs in cluster run
export N_NODES=31 # number of nodes over the MPC horizons
export CLUSTER_DB=1 # enable some db features for the cluster
export PHYSICS_DT=0.0005 # integration dt for the simulator
export USE_GPU_SIM=1 # whether to launch sim with GPU support (env has to read and implement this feature)
# export CODEGEN_OVERRIDE_BDIR="none"
export CODEGEN_OVERRIDE_BDIR="${HOME}/aux_data/MyRobotClusterClient_${SHM_NS}/CodeGen/${SHM_NS}Rhc" # the cluster will look here for previously codegenerated files
# export TRAIN_ENV_FNAME="twist_tracking_env"
# export TRAIN_ENV_CNAME="TwistTrackingEnv"
export TRAIN_ENV_FNAME="fake_pos_tracking_env" # training env to be loaded in AugMPC
export TRAIN_ENV_CNAME="FakePosTrackingEnv" # name of  traininv class in env file
# export TRAIN_ENV_FNAME="fake_pos_track_env_phase_control"
# export TRAIN_ENV_CNAME="FakePosTrackEnvPhaseControl"
# export TRAIN_ENV_FNAME="fake_pos_track_env_phase_control_with_demo"
# export TRAIN_ENV_CNAME="FakePosTrackEnvPhaseControlWithDemo"
# export TRAIN_ENV_FNAME="fake_pos_tracking_with_demo"
# export TRAIN_ENV_CNAME="FakePosTrackingEnvWithDemo"
# export TRAIN_ENV_FNAME="linvel_env_with_demo"
# export TRAIN_ENV_CNAME="TwistTrackingEnvWithDemo"
# export TRAIN_ENV_FNAME="flight_phase_control_env"
# export TRAIN_ENV_CNAME="FlightPhaseControl"
export BAG_SDT=90.0 # when recording bags while training, run for this amount of seconds
export BRIDGE_DT=0.05 # rate at which data on shared mem is read and published through ROS
export DUMP_DT=50.0 # lauch a bag every DUMP_DT [min]
export ENV_IDX_BAG=5 # environemnt index from which data will be recorded
export ENV_IDX_BAG_DEMO=-1 # if using demo envs, can additionally record a bag from demo env
export ENV_IDX_BAG_EXPL=-1 # same, but for exploration environments
export SRDF_PATH_ROSBAG="${HOME}/aux_data/KyonRHClusterClient_${SHM_NS}/$SHM_NS.srdf" # base path where the description package for the robot are located

export SET_ULIM=1 # whether to set maximum number of open file descriptors for each process (shared memory)
export ULIM_N=28672 # maximum number (might be necessary to increase due to heave shared memory usage by the framework and the
# control cluster specifically, when the number of environments is large)

export TIMEOUT_MS=120000 # timeout after which the simulator, cluster or training env autokills if no signal is received ([ms])

# custom configuration arguments that either the cluster, the simulator or the training environment accept (they are passed to all, and ignored if not necessary)
# should specify the name of the arg, the data type (bool, int, str, float, xacro) and its corresponding value. They will be parsed accordingly
export CUSTOM_ARGS_NAMES="step_height wheels fixed_flights adaptive_is lin_a_feedback closed_partial use_flat_ground estimate_v_root use_jnt_v_feedback base_linkname self_collide" 
export CUSTOM_ARGS_DTYPE="float xacro bool bool bool bool bool bool bool str bool"
export CUSTOM_ARGS_VALS="0.18 false true true false true true false false pelvis false" 
