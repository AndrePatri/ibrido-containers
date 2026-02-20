#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
AugMPC_DIR="$WS_ROOT/src/AugMPC/aug_mpc/scripts"
AugMPCEnvs_DIR="$WS_ROOT/src/AugMPCEnvs/aug_mpc_envs/scripts"

# PID holders for launched processes
cluster_pid=""
train_pid=""
world_iface_pid=""
bag_pid=""
joy_pid=""

# -------------------------
# Cleanup logic (escalate only on 2nd+ signal)
# -------------------------
_in_cleanup=0

cleanup_graceful() {
    echo "launch_training.sh: graceful cleanup: sending SIGINT to child processes (if running)..."

    if [ -n "$world_iface_pid" ] && kill -0 "$world_iface_pid" 2>/dev/null; then
        echo "launch_training.sh: sending SIGINT to world interface PID $world_iface_pid"
        kill -INT "$world_iface_pid" 2>/dev/null || true
    else
        echo "launch_training.sh: no world_iface_pid to signal (or it's not running)."
    fi

    if [ -n "$joy_pid" ] && kill -0 "$joy_pid" 2>/dev/null; then
        echo "launch_training.sh: sending SIGKILL to joy PID $joy_pid"
        kill -KILL "$joy_pid" 2>/dev/null || true
    else
        echo "launch_training.sh: no joy_pid to signal (or it's not running)."
    fi

    if [ -n "$bag_pid" ] && kill -0 "$bag_pid" 2>/dev/null; then
        echo "launch_training.sh: sending SIGINT to rosbag PID $bag_pid"
        kill -INT "$bag_pid" 2>/dev/null || true
    else
        echo "launch_training.sh: no bag_pid to signal (or it's not running)."
    fi

    echo "launch_training.sh: waiting for launched processes to exit..."
    for pid in "$cluster_pid" "$train_pid" "$world_iface_pid" "$bag_pid" "$joy_pid"; do
        if [ -n "$pid" ]; then
            echo "launch_training.sh: waiting for PID $pid ..."
            wait "$pid" 2>/dev/null || true
        fi
    done

    echo "launch_training.sh: graceful cleanup done."
}

cleanup_hard() {
    echo "launch_training.sh: HARD cleanup: sending SIGKILL to child processes (if running)..."

    # With setsid, killing the process group is often what you want.
    # Try PGID kill first (negative PID), fall back to PID kill.
    for pid in "$cluster_pid" "$train_pid" "$world_iface_pid" "$bag_pid" "$joy_pid"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "launch_training.sh: sending SIGKILL to PID/PGID $pid"
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    echo "launch_training.sh: waiting for processes to exit after SIGKILL..."
    for pid in "$cluster_pid" "$train_pid" "$world_iface_pid" "$bag_pid" "$joy_pid"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    echo "launch_training.sh: hard cleanup done."
}

on_signal() {
    local sig="$1"

    # 2nd+ signal -> hard kill
    if [ "$_in_cleanup" -eq 1 ]; then
        echo "launch_training.sh: received $sig again during cleanup -> escalating"
        cleanup_hard
        # Conventional exit codes: 128 + signal number
        case "$sig" in
            INT)  exit 130 ;; # 128+2
            QUIT) exit 131 ;; # 128+3
            TERM) exit 143 ;; # 128+15
            *)    exit 128 ;;
        esac
    fi

    _in_cleanup=1
    echo "launch_training.sh: received $sig -> starting graceful cleanup"
    cleanup_graceful

    case "$sig" in
        INT)  exit 130 ;;
        TERM) exit 143 ;;
        QUIT) exit 131 ;;
        *)    exit 1 ;;
    esac
}

on_exit() {
    if [ "$_in_cleanup" -eq 0 ]; then
        _in_cleanup=1
        cleanup_graceful
    fi
}

# Traps: smooth signals + EXIT
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal QUIT' QUIT
trap 'on_exit' EXIT
# NOTE: SIGKILL cannot be trapped.

# -------------------------
# Original script body
# -------------------------

usage() {
  echo "Usage: $0
    [--cfg CFG] \
    [--unique_id UNIQUE_ID] \
    "
  exit 1
}

cfg_file_basepath="/root/ibrido_files/training_cfgs"
config_file="${cfg_file_basepath}/training_cfg.sh"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --unique_id) unique_id="$2"; shift ;;
    -cfg|--cfg) config_file="${cfg_file_basepath}/$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -f "$config_file" ]; then
    source "$config_file"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate ${MAMBA_ENV_NAME}

source $HOME/ibrido_ws/setup.bash

base_log_dir="${HOME}/ibrido_logs/ibrido_run_${unique_id}"
mkdir -p "$base_log_dir"
cp $config_file "${base_log_dir}/"

log_world="${base_log_dir}/ibrido_world_interface${RUN_NAME}_${unique_id}.log"
log_cluster="${base_log_dir}/ibrido_mpc_cluster_${RUN_NAME}_${unique_id}.log"
log_train="${base_log_dir}/ibrido_training_env_${RUN_NAME}_${unique_id}.log"
log_bag="${base_log_dir}/ibrido_rosbag_env_${RUN_NAME}_${unique_id}.log"
log_joy="${base_log_dir}/ibrido_joy_cmds_${RUN_NAME}_${unique_id}.log"

echo "
launch_training.sh: logging output to->
world interface: $log_world
MPC cluster: $log_cluster
training env: $log_train
log bag: $log_bag
joy cmds: $log_joy
"

if (( $SET_ULIM )); then
  ulimit -n $ULIM_N
fi

SHM_NS+="${unique_id}"
echo "Will use shared memory namespace ${SHM_NS}"

cluster_cmd="--ns $SHM_NS --size $N_ENVS --timeout_ms $TIMEOUT_MS \
--codegen_override_dir $CODEGEN_OVERRIDE_BDIR \
--urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--cluster_dt $CLUSTER_DT \
--n_nodes $N_NODES \
--set_affinity "
if (( $CLUSTER_DB )); then
  cluster_cmd+="--enable_debug "
fi
if (( $IS_CLOSED_LOOP )); then
  cluster_cmd+="--cloop "
fi

setsid python $AugMPC_DIR/launch_control_cluster.py $cluster_cmd > "$log_cluster" 2>&1 &
cluster_pid=$!

if (( $REMOTE_STEPPING )); then
  training_env_cmd="--dump_checkpoints --ns $SHM_NS --drop_dir $HOME/training_data \
--db --env_db \
--step_while_setup \
--seed $SEED --timeout_ms $TIMEOUT_MS \
--env_fname $TRAIN_ENV_FNAME --env_classname $TRAIN_ENV_CNAME \
--demo_stop_thresh $DEMO_STOP_THRESH  \
--actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH \
--critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH \
--tot_tsteps $TOT_STEPS \
--demo_envs_perc $DEMO_ENVS_PERC \
--expl_envs_perc $EXPL_ENVS_PERC \
--action_repeat $ACTION_REPEAT \
--compression_ratio $COMPRESSION_RATIO "
  if (( $USE_DUMMY )); then
    training_env_cmd+="--dummy "
  elif (( $USE_SAC )); then
    training_env_cmd+="--sac "
  fi
  if (( $DUMP_ENV_CHECKPOINTS )); then
    training_env_cmd+="--full_env_db "
  fi
  if (( $OBS_NORM )); then
    training_env_cmd+="--obs_norm "
  fi
  if (( $OBS_RESCALING )); then
    training_env_cmd+="--obs_rescale "
  fi
  if (( $WEIGHT_NORM )); then
    training_env_cmd+="--add_weight_norm "
  fi
  if [[ -n "$RNAME" ]]; then
      training_env_cmd+="--run_name ${RNAME}_${TRAIN_ENV_CNAME} "
  fi
  if (( $EVAL )); then
    training_env_cmd+="--eval --n_eval_timesteps $TOT_STEPS --mpath $MPATH --mname $MNAME "
    if (( $DET_EVAL )); then
      training_env_cmd+="--det_eval "
    fi
    if (( $EVAL_ON_CPU )); then
      training_env_cmd+="--use_cpu "
    fi
    if (( $OVERRIDE_ENV )); then
      training_env_cmd+="--override_env "
    fi
    if (( $OVERRIDE_AGENT_REFS )); then
      training_env_cmd+="--override_env "
    fi
    if (( $OVERRIDE_AGENT_REFS )); then
      training_env_cmd+="--override_agent_refs "
    fi
  fi

  setsid python $AugMPC_DIR/launch_train_env.py $training_env_cmd --comment "\"$COMMENT\"" > "$log_train" 2>&1 &
  train_pid=$!
fi

source /opt/ros/noetic/setup.bash
source /opt/xbot/setup.sh
source $HOME/ibrido_ws/setup.bash

remote_env_cmd="--robot_name $SHM_NS \
--urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
--jnt_imp_config_path $JNT_IMP_CF_PATH \
--world_iface_fname $REMOTE_ENV_FNAME \
--cluster_dt $CLUSTER_DT \
--num_envs $N_ENVS --seed $SEED --timeout_ms $TIMEOUT_MS \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--physics_dt $PHYSICS_DT \
--enable_debug "
if (( $REMOTE_STEPPING )); then
  remote_env_cmd+="--remote_stepping "
fi

python $AugMPC_DIR/launch_world_interface.py $remote_env_cmd > "$log_world" 2>&1 &
world_iface_pid=$!

if (( $LAUNCH_JOY )); then
  if (( $AGENT_JOY )); then
    if (( $XBOT2_JOY )); then
      joy_cmd="--ns $SHM_NS --env_idx 0 --agent_refs --mode $JOY_MODE"
      cmd="$AugMPCEnvs_DIR/launch_xbot2_joy_cmds.py $joy_cmd"
    else
      joy_cmd="--ns $SHM_NS --env_idx 0 --agent_refs_world --add_remote_exit"
      cmd="$AugMPC_DIR/utilities/launch_agent_keybrd_cmds.py $joy_cmd"
    fi
  else
    if (( $XBOT2_JOY )); then
      joy_cmd="--ns $SHM_NS --env_idx 0"
      cmd="$AugMPCEnvs_DIR/launch_xbot2_joy_cmds.py $joy_cmd"
    else
      joy_cmd="--ns $SHM_NS --env_idx 0 --joy --add_remote_exit"
      cmd="$AugMPC_DIR/utilities/launch_rhc_keybrd_cmds.py $joy_cmd"
    fi
  fi
  python $cmd > "$log_joy" 2>&1 &
  joy_pid=$!
fi

# Wait for all launched processes (keeps script alive)
for pid in "$cluster_pid" "$train_pid" "$world_iface_pid" "$bag_pid" "$joy_pid"; do
  if [ -n "$pid" ]; then
    wait "$pid" 2>/dev/null || true
  fi
done