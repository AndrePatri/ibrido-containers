#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/AugMPC/aug_mpc/scripts"
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${UTILS_DIR}/ibrido_command_builder.sh"

child_pids=()
cleanup_started=0

signal_process_tree() {
    local signal="$1"
    shift

    local pid child_pids child_pid
    for pid in "$@"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        child_pids="$(pgrep -P "$pid" 2>/dev/null || true)"
        for child_pid in $child_pids; do
            signal_process_tree "$signal" "$child_pid"
        done

        kill "-${signal}" "$pid" 2>/dev/null || true
    done
}

wait_for_children_exit() {
    local timeout_s="$1"
    local elapsed_s=0
    local child_pid any_alive

    while (( elapsed_s < timeout_s )); do
        any_alive=0
        for child_pid in "${child_pids[@]}"; do
            if kill -0 "$child_pid" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        if (( ! any_alive )); then
            return 0
        fi
        sleep 1
        elapsed_s=$((elapsed_s + 1))
    done

    return 1
}

cleanup() {
    if (( cleanup_started )); then
        return
    fi
    cleanup_started=1

    echo "launch_training.sh: sending SIGINT to all child processes..."
    if ((${#child_pids[@]})); then
        signal_process_tree INT "${child_pids[@]}"
        if ! wait_for_children_exit 8; then
            echo "launch_training.sh: children still alive, sending SIGTERM..."
            signal_process_tree TERM "${child_pids[@]}"
        fi
        if ! wait_for_children_exit 8; then
            echo "launch_training.sh: children still alive, sending SIGKILL..."
            signal_process_tree KILL "${child_pids[@]}"
        fi
    fi

    echo "launch_training.sh: waiting for child processes to exit..."
    for child_pid in "${child_pids[@]}"; do
        wait "$child_pid" 2>/dev/null
    done

    echo "launch_training.sh: all child processes have exited."
}

# Trap EXIT, INT (Ctrl+C), and TERM signals to trigger cleanup
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

usage() {
  echo "Usage: $0
    --cfg CFG \
    [--unique_id UNIQUE_ID] \
    [--set VAR=VALUE] \
    [--dry-run] \
    "
  exit 1
}

cfg_file_basepath="${IBRIDO_CFG_BASEPATH:-/root/ibrido_files/training_cfgs}"

config_file=""
dry_run=0
cfg_overrides=()

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --unique_id) unique_id="$2"; shift ;;
    -cfg|--cfg)
      if [[ "$2" = /* ]]; then
        config_file="$2"
      else
        config_file="${cfg_file_basepath}/$2"
      fi
      shift
      ;;
    --set) cfg_overrides+=("$2"); shift ;;
    --dry-run) dry_run=1 ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

unique_id="${unique_id:-$(date '+%Y_%m_%d__%H_%M_%S')}"

if [ -z "$config_file" ]; then
    echo "launch_training.sh: --cfg is required"
    usage
fi

# Load the declarative configuration file.
if [ -f "$config_file" ]; then
    config_exports="$("${UTILS_DIR}/ibrido_config_loader.py" --shell "$config_file")" || exit 1
    eval "$config_exports"
else
    echo "Configuration file not found: $config_file"
    exit 1
fi

for cfg_override in "${cfg_overrides[@]}"; do
  cfg_override_name="${cfg_override%%=*}"
  if [[ "$cfg_override" != *=* || ! "$cfg_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "Invalid --set override: $cfg_override"
    exit 1
  fi
  export "$cfg_override"
done

ibrido_normalize_runtime_config || exit 1
ibrido_validate_runtime_config || exit 1

# clear tmp folder
# rm -r /tmp/*

# activate micromamba for this shell
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
elif (( ! dry_run )); then
  echo "launch_training.sh: micromamba is not available on PATH"
  exit 1
fi

#!/bin/bash

# Define the timestamp and log file based on RUN_NAME and unique id
base_log_dir="${HOME}/ibrido_logs/ibrido_run_${unique_id}"
mkdir -p "$base_log_dir"
cp "$config_file" "${base_log_dir}/"

run_label="${RUN_NAME:-${RNAME:-IbridoRun}}"
log_world="${base_log_dir}/ibrido_world_interface_${run_label}_${unique_id}.log"
log_cluster="${base_log_dir}/ibrido_mpc_cluster_${run_label}_${unique_id}.log"
log_train="${base_log_dir}/ibrido_training_env_${run_label}_${unique_id}.log"
log_bag="${base_log_dir}/ibrido_rosbag_env_${run_label}_${unique_id}.log"

# Ensure the log directory exists

# # Redirect stdout and stderr to both the terminal and the log file
echo "
launch_training.sh: logging output to->
remote env: $log_world
rhc cluster: $log_cluster
train. env: $log_train
log bag: $log_bag
"

if (( $SET_ULIM )); then
  ulimit -n $ULIM_N
fi

SHM_NS+="${unique_id}" # appending unique string to actual shm namespace
echo "Will use shared memory namespace ${SHM_NS}"

export IBRIDO_RUN_META_DIR="${IBRIDO_RUN_META_DIR:-${base_log_dir}/metadata}"

world_iface_fname="${WORLD_INTERFACE:-}"
world_headless="${WORLD_HEADLESS:-1}"
world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
world_use_gpu="${USE_GPU_SIM:-1}"
world_env_profile="isaac5x"
world_jnt_imp_config_path="${JNT_IMP_CONFIG_PATH:-}"

if [ -z "$world_iface_fname" ]; then
  echo "launch_training.sh: WORLD_INTERFACE is required by the selected cfg"
  exit 1
fi

case "$world_iface_fname" in
  *xmj_world_interface*)
    world_headless="${XMJ_HEADLESS:-0}"
    world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-0}"
    world_use_gpu=0
    world_env_profile="xbot"
    # the world interface generates URDF/SRDF and the runtime XBot config itself
    # (from the template xbot config + template impedance); pass templates only
    ;;
  *rt_deploy_world_interface*)
    world_headless="${RT_HEADLESS:-0}"
    world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-0}"
    world_use_gpu=0
    world_env_profile="xbot"
    # the world interface generates URDF/SRDF and the runtime XBot config itself
    # (from the template xbot config + template impedance); pass templates only
    ;;
  *isaac5x_world_interface*)
    world_headless="${WORLD_HEADLESS:-1}"
    world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
    world_use_gpu="${USE_GPU_SIM:-1}"
    world_env_profile="isaac5x"
    ;;
  *)
    echo "launch_training.sh: unsupported WORLD_INTERFACE: $world_iface_fname"
    exit 1
    ;;
esac

export WORLD_INTERFACE="$world_iface_fname"
ibrido_build_world_cmd "$world_iface_fname" "$N_ENVS" "$world_headless" "$world_use_custom_jnt_imp" "$world_use_gpu" "$world_jnt_imp_config_path"
remote_env_cmd="$IBRIDO_WORLD_CMD"

ibrido_build_cluster_cmd "$N_ENVS" 1 0
cluster_cmd="$IBRIDO_CLUSTER_CMD"

training_env_cmd=""
if (( $REMOTE_STEPPING )); then
  ibrido_build_training_cmd
  training_env_cmd="$IBRIDO_TRAINING_CMD"
fi

world_launch_cmd="python $LRHC_DIR/launch_world_interface.py $remote_env_cmd"
cluster_launch_cmd="python $LRHC_DIR/launch_control_cluster.py $cluster_cmd"
training_launch_cmd=""
if [ -n "$training_env_cmd" ]; then
  training_launch_cmd="python $LRHC_DIR/launch_train_env.py $training_env_cmd --comment \"\\\"$COMMENT\\\"\""
fi

ibrido_prepare_run_metadata "$config_file" "$world_launch_cmd" "$cluster_launch_cmd" "$training_launch_cmd" "$unique_id"

if (( dry_run )); then
  ibrido_print_dry_run "$world_launch_cmd" "$cluster_launch_cmd" "$training_launch_cmd"
  trap - EXIT INT TERM
  exit 0
fi

if [ "$world_env_profile" = "isaac5x" ]; then
  (
    micromamba activate "${MAMBA_ENV_NAME_ISAAC:-ibrido_isaac_py11}"
    source /isaac-sim/setup_conda_env.sh
    source "$HOME/ibrido_ws/setup.bash"
    export EXP_PATH="/isaac-sim/apps"
    exec python "$LRHC_DIR/launch_world_interface.py" $remote_env_cmd
  ) > "$log_world" 2>&1 &
else
  (
    micromamba activate "${MAMBA_ENV_NAME:-ibrido}"
    source "$HOME/ibrido_ws/setup.bash"
    exec python "$LRHC_DIR/launch_world_interface.py" $remote_env_cmd
  ) > "$log_world" 2>&1 &
fi
world_pid=$!
child_pids+=("$world_pid")

# cluster
(
  micromamba activate ${MAMBA_ENV_NAME}
  source $HOME/ibrido_ws/setup.bash
  exec python $LRHC_DIR/launch_control_cluster.py $cluster_cmd
) > "$log_cluster" 2>&1 &
cluster_pid=$!
child_pids+=("$cluster_pid")

# train env
if (( $REMOTE_STEPPING )); then

(
  micromamba activate ${MAMBA_ENV_NAME}
  source $HOME/ibrido_ws/setup.bash
  if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_KEY="${WANDB_KEY:-$WANDB_API_KEY}"
  elif [ -n "${WANDB_KEY:-}" ]; then
    export WANDB_API_KEY="$WANDB_KEY"
  fi
  if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
  fi
  exec python $LRHC_DIR/launch_train_env.py $training_env_cmd --comment "\"$COMMENT\""
) > "$log_train" 2>&1 &
training_env_pid=$!
child_pids+=("$training_env_pid")
fi

# rosbag db
if (( $ENV_IDX_BAG >= 0 && $CLUSTER_DB)); then
  rosbag_cmd="--ros2 --use_shared_drop_dir --pub_stime --is_training \
  --ns $SHM_NS --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG "
  if (( $PUB_HEIGHTMAP )); then
    rosbag_cmd+="--show_heightmap "
  fi
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs --no_rhc_internal "
  fi
  (
    micromamba activate ${MAMBA_ENV_NAME}
    source /opt/ros/jazzy/setup.bash
    source $HOME/ibrido_ws/setup.bash
    exec python $LRHC_DIR/utilities/launch_periodic_bag_dump.py $rosbag_cmd
  ) > "$log_bag" 2>&1 &
  child_pids+=("$!")
fi

# demo env db
sleep 2 # wait a bit
if (( $ENV_IDX_BAG_DEMO >= 0 && $CLUSTER_DB)); then
  rosbag_cmd="--ros2 --use_shared_drop_dir --is_training \
  --ns $SHM_NS --remap_ns "${SHM_NS}_demo" \
  --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG_DEMO "
  if (( $PUB_HEIGHTMAP )); then
    rosbag_cmd+="--show_heightmap "
  fi
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs --no_rhc_internal "
  fi
  (
    micromamba activate ${MAMBA_ENV_NAME}
    source /opt/ros/jazzy/setup.bash
    source $HOME/ibrido_ws/setup.bash
    exec python $LRHC_DIR/utilities/launch_periodic_bag_dump.py $rosbag_cmd
  ) > "$log_bag" 2>&1 &
  child_pids+=("$!")
fi

# expl env db
sleep 2 # wait a bit
if (( $ENV_IDX_BAG_EXPL >= 0 && $CLUSTER_DB)); then
  rosbag_cmd="--ros2 --use_shared_drop_dir --is_training \
  --ns $SHM_NS --remap_ns "${SHM_NS}_expl" \
  --rhc_refs_in_h_frame \
  --srdf_path $SRDF_PATH_ROSBAG \
  --bag_sdt $BAG_SDT --ros_bridge_dt $BRIDGE_DT --dump_dt_min $DUMP_DT --env_idx $ENV_IDX_BAG_EXPL "
  if (( $PUB_HEIGHTMAP )); then
  rosbag_cmd+="--show_heightmap "
  fi
  if (( $REMOTE_STEPPING )); then
  rosbag_cmd+="--with_agent_refs --no_rhc_internal "
  fi
  (
    micromamba activate ${MAMBA_ENV_NAME}
    source /opt/ros/jazzy/setup.bash
    source $HOME/ibrido_ws/setup.bash
    exec python $LRHC_DIR/utilities/launch_periodic_bag_dump.py $rosbag_cmd
  ) > "$log_bag" 2>&1 &
  child_pids+=("$!")
fi

if (( $REMOTE_STEPPING )); then
  wait "$training_env_pid"
  training_env_status=$?
  cleanup
  trap - EXIT INT TERM
  exit "$training_env_status"
fi

wait "${child_pids[@]}"
