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

    if ((${#child_pids[@]} == 0)); then
        return
    fi

    echo "launch_bundle.sh: sending SIGINT to all child processes..."
    if ((${#child_pids[@]})); then
        signal_process_tree INT "${child_pids[@]}"
        if ! wait_for_children_exit 8; then
            echo "launch_bundle.sh: children still alive, sending SIGTERM..."
            signal_process_tree TERM "${child_pids[@]}"
        fi
        if ! wait_for_children_exit 8; then
            echo "launch_bundle.sh: children still alive, sending SIGKILL..."
            signal_process_tree KILL "${child_pids[@]}"
        fi
    fi

    echo "launch_bundle.sh: waiting for child processes to exit..."
    for child_pid in "${child_pids[@]}"; do
        wait "$child_pid" 2>/dev/null
    done

    echo "launch_bundle.sh: all child processes have exited."
}

trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

usage() {
    echo "Usage: $0
    --bundle BUNDLE_DIR_OR_BUNDLE_YAML \
    [--intent eval_same_domain] \
    [--unique_id UNIQUE_ID] \
    [--set VAR=VALUE] \
    [--allow_contract_override] \
    [--dry-run] \
    "
    exit 1
}

yaml_top_scalar() {
    local file="$1"
    local key="$2"
    local value

    value="$(sed -n "s/^${key}:[[:space:]]*//p" "$file" | head -n 1)"
    value="${value%%#*}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s' "$value"
}

sanitize_token() {
    local token="$1"
    token="${token//[^A-Za-z0-9_]/_}"
    printf '%s' "$token"
}

is_same_domain_contract_var() {
    case "$1" in
        RUN_INTENT|WORLD_BACKEND|WORLD_INTERFACE|REMOTE_ENV_FNAME|IBRIDO_WORLD_INTERFACE|\
        URDF_PATH|SRDF_PATH|JNT_IMP_CONFIG_PATH|CLUSTER_CL_FNAME|CLUSTER_DT|PHYSICS_DT|\
        N_NODES|TRAIN_ENV_FNAME|TRAIN_ENV_CNAME|ACTION_REPEAT|CUSTOM_ARGS_NAMES|\
        CUSTOM_ARGS_DTYPE|CUSTOM_ARGS_VALS)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

bundle_path=""
intent="eval_same_domain"
unique_id=""
dry_run=0
allow_contract_override=0
cfg_overrides=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bundle) bundle_path="$2"; shift ;;
        --intent) intent="$2"; shift ;;
        --unique_id) unique_id="$2"; shift ;;
        --set) cfg_overrides+=("$2"); shift ;;
        --allow_contract_override) allow_contract_override=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$bundle_path" ]; then
    echo "launch_bundle.sh: --bundle is required"
    usage
fi

if [ "$intent" != "eval_same_domain" ]; then
    echo "launch_bundle.sh: unsupported intent '$intent'. Only eval_same_domain is wired for now."
    exit 1
fi

if [ -f "$bundle_path" ]; then
    bundle_file="$(readlink -f "$bundle_path")"
    bundle_dir="$(dirname "$bundle_file")"
else
    bundle_dir="$(readlink -f "$bundle_path")"
    bundle_file="${bundle_dir}/bundle.yaml"
fi

if [ ! -f "$bundle_file" ]; then
    echo "launch_bundle.sh: bundle.yaml not found: $bundle_file"
    exit 1
fi

checkpoint_file="$(yaml_top_scalar "$bundle_file" checkpoint_file)"
run_metadata_rel="$(yaml_top_scalar "$bundle_file" run_metadata)"
resolved_config_rel="$(yaml_top_scalar "$bundle_file" resolved_config)"

if [ -z "$checkpoint_file" ]; then
    echo "launch_bundle.sh: checkpoint_file is missing from $bundle_file"
    exit 1
fi

source_run_manifest="${bundle_dir}/${run_metadata_rel:-run_metadata/run_manifest.yaml}"
source_run_metadata_dir="$(dirname "$source_run_manifest")"
source_resolved_config="${bundle_dir}/${resolved_config_rel:-run_metadata/resolved_config.yaml}"
source_resolved_env="${source_run_metadata_dir}/resolved_env.sh"

if [ ! -f "$source_resolved_env" ]; then
    echo "launch_bundle.sh: source resolved env not found: $source_resolved_env"
    exit 1
fi

# Restore the training-time config, then intentionally override only the eval layer.
source "$source_resolved_env"
unset IBRIDO_RUN_META_DIR

unique_id="${unique_id:-$(date '+%Y_%m_%d__%H_%M_%S')}"
bundle_name="$(basename "$bundle_dir")"
bundle_token="$(sanitize_token "$bundle_name")"
run_label="${RNAME:-${RUN_NAME:-BundleRun}}"

export SOURCE_BUNDLE_PATH="$bundle_dir"
export SOURCE_BUNDLE_FILE="$bundle_file"
export SOURCE_RUN_METADATA_DIR="$source_run_metadata_dir"
export SOURCE_RESOLVED_CONFIG_PATH="$source_resolved_config"

export RUN_INTENT="eval_same_domain"
export WORLD_BACKEND="isaac5x"
export TIME_SOURCE="sim"
export EVAL="1"
export RESUME="0"
export REMOTE_STEPPING="1"
export DET_EVAL="${DET_EVAL:-1}"
export EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
export OVERRIDE_ENV="${OVERRIDE_ENV:-0}"
export OVERRIDE_AGENT_REFS="${OVERRIDE_AGENT_REFS:-0}"
export DEBUG="0"
export RMDEBUG="0"
export DUMP_ENV_CHECKPOINTS="0"
export CLUSTER_DB="0"
export ENV_IDX_BAG="-1"
export ENV_IDX_BAG_DEMO="-1"
export ENV_IDX_BAG_EXPL="-1"
export N_ENVS="${EVAL_N_ENVS:-2}"
export TOT_STEPS="${EVAL_TOT_STEPS:-1000}"
export MPATH="$bundle_dir"
export MNAME="$checkpoint_file"
export RNAME="${run_label}_EvalSameDomain"
export COMMENT="eval_same_domain from ${bundle_name}"
export SHM_NS="${SHM_NS_EVAL_BASE:-eval_${bundle_token}}${unique_id}"

for cfg_override in "${cfg_overrides[@]}"; do
    cfg_override_name="${cfg_override%%=*}"
    if [[ "$cfg_override" != *=* || ! "$cfg_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid --set override: $cfg_override"
        exit 1
    fi
    if (( ! allow_contract_override )) && is_same_domain_contract_var "$cfg_override_name"; then
        echo "launch_bundle.sh: refusing same-domain contract override '$cfg_override_name'."
        echo "launch_bundle.sh: pass --allow_contract_override if this mismatch is intentional."
        exit 1
    fi
    export "$cfg_override"
done

ibrido_normalize_runtime_config || exit 1
ibrido_validate_runtime_config || exit 1

if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
elif (( ! dry_run )); then
    echo "launch_bundle.sh: micromamba is not available on PATH"
    exit 1
fi

base_log_dir="${HOME}/ibrido_logs/ibrido_eval_${unique_id}"
mkdir -p "$base_log_dir"
cp "$bundle_file" "${base_log_dir}/"

log_world="${base_log_dir}/ibrido_world_interface_${run_label}_${unique_id}.log"
log_cluster="${base_log_dir}/ibrido_mpc_cluster_${run_label}_${unique_id}.log"
log_eval="${base_log_dir}/ibrido_eval_env_${run_label}_${unique_id}.log"

echo "
launch_bundle.sh: logging output to->
source bundle: $bundle_file
remote env: $log_world
rhc cluster: $log_cluster
eval env: $log_eval
"

if (( ${SET_ULIM:-0} )); then
    ulimit -n "${ULIM_N:-131072}"
fi

echo "Will use shared memory namespace ${SHM_NS}"

export IBRIDO_RUN_META_DIR="${base_log_dir}/metadata"

ibrido_build_world_cmd "aug_mpc_envs.world_interfaces.isaac5x_world_interface" "$N_ENVS" 1 1 "$USE_GPU_SIM"
remote_env_cmd="$IBRIDO_WORLD_CMD"

ibrido_build_cluster_cmd "$N_ENVS" 1 0
cluster_cmd="$IBRIDO_CLUSTER_CMD"

ibrido_build_training_cmd
eval_env_cmd="$IBRIDO_TRAINING_CMD"

world_launch_cmd="python $LRHC_DIR/launch_world_interface.py $remote_env_cmd"
cluster_launch_cmd="python $LRHC_DIR/launch_control_cluster.py $cluster_cmd"
eval_launch_cmd="python $LRHC_DIR/launch_train_env.py $eval_env_cmd --comment \"\\\"$COMMENT\\\"\""

ibrido_prepare_run_metadata "$bundle_file" "$world_launch_cmd" "$cluster_launch_cmd" "$eval_launch_cmd" "$unique_id"

if (( dry_run )); then
    ibrido_print_dry_run "$world_launch_cmd" "$cluster_launch_cmd" "$eval_launch_cmd"
    trap - EXIT INT TERM
    exit 0
fi

(
    micromamba activate "${MAMBA_ENV_NAME_ISAAC:-ibrido_isaac_py11}"
    source /isaac-sim/setup_conda_env.sh
    source "$HOME/ibrido_ws/setup.bash"
    export EXP_PATH="/isaac-sim/apps"
    exec python "$LRHC_DIR/launch_world_interface.py" $remote_env_cmd
) > "$log_world" 2>&1 &
world_pid=$!
child_pids+=("$world_pid")

(
    micromamba activate "${MAMBA_ENV_NAME:-ibrido}"
    source "$HOME/ibrido_ws/setup.bash"
    exec python "$LRHC_DIR/launch_control_cluster.py" $cluster_cmd
) > "$log_cluster" 2>&1 &
cluster_pid=$!
child_pids+=("$cluster_pid")

(
    micromamba activate "${MAMBA_ENV_NAME:-ibrido}"
    source "$HOME/ibrido_ws/setup.bash"
    exec python "$LRHC_DIR/launch_train_env.py" $eval_env_cmd --comment "\"$COMMENT\""
) > "$log_eval" 2>&1 &
eval_env_pid=$!
child_pids+=("$eval_env_pid")

wait "$eval_env_pid"
eval_env_status=$?
cleanup
trap - EXIT INT TERM
exit "$eval_env_status"
