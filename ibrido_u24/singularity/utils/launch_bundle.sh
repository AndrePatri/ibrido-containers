#!/bin/bash

WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/AugMPC/aug_mpc/scripts"
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${UTILS_DIR}/ibrido_command_builder.sh"

child_pids=()
cleanup_started=0
cleanup_completed=0
shutdown_stage=0
primary_pid=""

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

wait_for_pid() {
    local pid="$1"
    local status=0

    while kill -0 "$pid" 2>/dev/null; do
        wait "$pid" 2>/dev/null
        status=$?
    done

    return "$status"
}

wait_for_children() {
    local child_pid
    for child_pid in "${child_pids[@]}"; do
        wait_for_pid "$child_pid" || true
    done
}

force_shutdown() {
    if (( shutdown_stage < 2 )); then
        shutdown_stage=2
        echo "launch_bundle.sh: forcing shutdown with SIGTERM..."
        signal_process_tree TERM "${child_pids[@]}"
    else
        shutdown_stage=3
        echo "launch_bundle.sh: forcing shutdown with SIGKILL..."
        signal_process_tree KILL "${child_pids[@]}"
    fi
}

request_shutdown() {
    if (( shutdown_stage == 0 )); then
        shutdown_stage=1
        if [ -n "$primary_pid" ] && kill -0 "$primary_pid" 2>/dev/null; then
            echo "launch_bundle.sh: requesting graceful evaluation shutdown; press Ctrl-C again to force it..."
            signal_process_tree INT "$primary_pid"
        else
            echo "launch_bundle.sh: requesting graceful shutdown; press Ctrl-C again to force it..."
            signal_process_tree INT "${child_pids[@]}"
        fi
    else
        force_shutdown
    fi
}

cleanup() {
    if (( cleanup_completed )); then
        return
    fi

    if (( ! cleanup_started )); then
        cleanup_started=1
        echo "launch_bundle.sh: evaluation stopped; sending SIGINT to remaining child processes..."
        signal_process_tree INT "${child_pids[@]}"
    fi

    echo "launch_bundle.sh: waiting for child processes to exit..."
    wait_for_children

    cleanup_completed=1
    echo "launch_bundle.sh: all child processes have exited."
}

trap cleanup EXIT
trap request_shutdown INT
trap force_shutdown TERM

usage() {
    echo "Usage: $0
    --bundle BUNDLE_DIR_OR_BUNDLE_YAML \
    --cfg TRANSFER_CFG \
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

resolve_ws_src_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s' "$path"
    else
        printf '%s/src/%s' "$WS_ROOT" "$path"
    fi
}

custom_args_has() {
    local name="$1"
    [[ " ${CUSTOM_ARGS_NAMES:-} " == *" ${name} "* ]]
}

append_custom_arg() {
    local name="$1"
    local dtype="$2"
    local value="$3"

    if custom_args_has "$name"; then
        return 0
    fi

    if [ -z "${CUSTOM_ARGS_NAMES:-}" ]; then
        CUSTOM_ARGS_NAMES="$name"
        CUSTOM_ARGS_DTYPE="$dtype"
        CUSTOM_ARGS_VALS="$value"
    else
        CUSTOM_ARGS_NAMES="${CUSTOM_ARGS_NAMES} ${name}"
        CUSTOM_ARGS_DTYPE="${CUSTOM_ARGS_DTYPE} ${dtype}"
        CUSTOM_ARGS_VALS="${CUSTOM_ARGS_VALS} ${value}"
    fi
    export CUSTOM_ARGS_NAMES CUSTOM_ARGS_DTYPE CUSTOM_ARGS_VALS
}

append_custom_args_triplets() {
    local names_s="$1"
    local dtype_s="$2"
    local vals_s="$3"
    local names=()
    local dtypes=()
    local vals=()
    local i

    read -r -a names <<< "$names_s"
    read -r -a dtypes <<< "$dtype_s"
    read -r -a vals <<< "$vals_s"

    if [ "${#names[@]}" -ne "${#dtypes[@]}" ] || [ "${#names[@]}" -ne "${#vals[@]}" ]; then
        echo "launch_bundle.sh: transfer custom args triplet length mismatch"
        exit 1
    fi

    for i in "${!names[@]}"; do
        append_custom_arg "${names[$i]}" "${dtypes[$i]}" "${vals[$i]}"
    done
}

resolve_xbot_config() {
    local cfg_key

    RESOLVED_XBOT_CONFIG_PATH="${XBOT_CONFIG_PATH:-}"
    if [ -n "$RESOLVED_XBOT_CONFIG_PATH" ]; then
        return
    fi

    RESOLVED_XBOT_CONFIG="${XBOT_CONFIG:-}"
    if [ -n "$RESOLVED_XBOT_CONFIG" ]; then
        RESOLVED_XBOT_CONFIG_PATH="$(resolve_ws_src_path "$RESOLVED_XBOT_CONFIG")"
        return
    fi

    cfg_key="${ROBOT_FAMILY:-} ${ROBOT_VARIANT:-} ${SHM_NS:-} ${RNAME:-} ${URDF_PATH:-} ${CLUSTER_CL_FNAME:-} ${CUSTOM_ARGS_VALS:-}"
    cfg_key="${cfg_key,,}"

    if [[ "$cfg_key" == *centauro* ]]; then
        RESOLVED_XBOT_CONFIG="CentauroHybridMPC/centaurohybridmpc/config/xmj_env_files/xbot2_basic.yaml"
    elif [[ "$cfg_key" == *b2w* ]]; then
        RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/b2w/xbot2_basic.yaml"
    elif [[ "$cfg_key" == *kyon* ]]; then
        if [[ "$cfg_key" == *iit-kyon-ros-pkg* || "$cfg_key" == *kyon_simple* ]]; then
            if [[ "$cfg_key" == *wheels* && "$cfg_key" != *no_wheels* ]]; then
                RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic_wheels.yaml"
            else
                RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/xbot2_basic.yaml"
            fi
        elif [[ "$cfg_key" == *wheels_no_yaw* || "$cfg_key" == *no_yaw* ]]; then
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic_wheels_no_yaw.yaml"
        elif [[ "$cfg_key" == *wheels* && "$cfg_key" != *no_wheels* ]]; then
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic_wheels.yaml"
        else
            RESOLVED_XBOT_CONFIG="KyonRLStepping/kyonrlstepping/config/xmj_env_files/kyon_real/xbot2_basic.yaml"
        fi
    elif [[ "$cfg_key" == *talos* ]]; then
        RESOLVED_XBOT_CONFIG="TalosHybridMPC/taloshybridmpc/config/xmj_env_files/xbot2_basic.yaml"
    fi

    if [ -n "$RESOLVED_XBOT_CONFIG" ]; then
        RESOLVED_XBOT_CONFIG_PATH="$(resolve_ws_src_path "$RESOLVED_XBOT_CONFIG")"
    fi
}

prepare_resolved_xbot_runtime_config() {
    local saved_ws_src="${IBRIDO_WS_SRC:-}"

    resolve_xbot_config
    if [ -z "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
        return
    fi

    export IBRIDO_WS_SRC="${HOME}/ibrido_ws/src"
    ibrido_prepare_xbot_runtime_config "$RESOLVED_XBOT_CONFIG_PATH" "$JNT_IMP_CONFIG_PATH" || exit 1
    RESOLVED_XBOT_TEMPLATE_CONFIG_PATH="$IBRIDO_XBOT_TEMPLATE_CONFIG_PATH"
    RESOLVED_XBOT_CONFIG_PATH="$IBRIDO_XBOT_RUNTIME_CONFIG_PATH"

    if [ -n "$saved_ws_src" ]; then
        export IBRIDO_WS_SRC="$saved_ws_src"
    else
        unset IBRIDO_WS_SRC
    fi
}

resolve_xmj_files_dir() {
    RESOLVED_XMJ_FILES_DIR_PATH="${XMJ_FILES_DIR_PATH:-}"
    if [ -n "$RESOLVED_XMJ_FILES_DIR_PATH" ]; then
        return
    fi

    RESOLVED_XMJ_FILES_DIR_PATH="${XMJ_FILES_DIR:-}"
    if [ -n "$RESOLVED_XMJ_FILES_DIR_PATH" ]; then
        RESOLVED_XMJ_FILES_DIR_PATH="$(resolve_ws_src_path "$RESOLVED_XMJ_FILES_DIR_PATH")"
        return
    fi

    resolve_xbot_config
    if [ -n "${RESOLVED_XBOT_CONFIG:-}" ]; then
        RESOLVED_XMJ_FILES_DIR_PATH="$(resolve_ws_src_path "${RESOLVED_XBOT_CONFIG%/*}")"
    fi
}

is_contract_var() {
    case "$1" in
        WORLD_INTERFACE|IBRIDO_WORLD_INTERFACE|\
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
cfg_file_basepath="${IBRIDO_CFG_BASEPATH:-/root/ibrido_files/training_cfgs}"
config_file=""
unique_id=""
dry_run=0
allow_contract_override=0
cfg_overrides=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bundle) bundle_path="$2"; shift ;;
        -cfg|--cfg)
            if [[ "$2" = /* ]]; then
                config_file="$2"
            else
                config_file="${cfg_file_basepath}/$2"
            fi
            shift
            ;;
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
if [ -z "$config_file" ]; then
    echo "launch_bundle.sh: --cfg is required"
    usage
fi
if [ ! -f "$config_file" ]; then
    echo "launch_bundle.sh: transfer cfg not found: $config_file"
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

runtime_home="$HOME"
runtime_path="$PATH"
runtime_ld_library_path_is_set=0
runtime_ld_library_path=""
if [ "${LD_LIBRARY_PATH+x}" = x ]; then
    runtime_ld_library_path_is_set=1
    runtime_ld_library_path="$LD_LIBRARY_PATH"
fi
runtime_pythonpath_is_set=0
runtime_pythonpath=""
if [ "${PYTHONPATH+x}" = x ]; then
    runtime_pythonpath_is_set=1
    runtime_pythonpath="$PYTHONPATH"
fi

source "$source_resolved_env"

export HOME="$runtime_home"
export PATH="$runtime_path"
if (( runtime_ld_library_path_is_set )); then
    export LD_LIBRARY_PATH="$runtime_ld_library_path"
else
    unset LD_LIBRARY_PATH
fi
if (( runtime_pythonpath_is_set )); then
    export PYTHONPATH="$runtime_pythonpath"
else
    unset PYTHONPATH
fi
unset IBRIDO_RUN_META_DIR

source_custom_args_names="${CUSTOM_ARGS_NAMES:-}"
source_custom_args_dtype="${CUSTOM_ARGS_DTYPE:-}"
source_custom_args_vals="${CUSTOM_ARGS_VALS:-}"

config_exports="$("${UTILS_DIR}/ibrido_config_loader.py" --shell "$config_file")" || exit 1
eval "$config_exports"

transfer_custom_args_names="${CUSTOM_ARGS_NAMES:-}"
transfer_custom_args_dtype="${CUSTOM_ARGS_DTYPE:-}"
transfer_custom_args_vals="${CUSTOM_ARGS_VALS:-}"
CUSTOM_ARGS_NAMES="$source_custom_args_names"
CUSTOM_ARGS_DTYPE="$source_custom_args_dtype"
CUSTOM_ARGS_VALS="$source_custom_args_vals"
export CUSTOM_ARGS_NAMES CUSTOM_ARGS_DTYPE CUSTOM_ARGS_VALS
append_custom_args_triplets "$transfer_custom_args_names" "$transfer_custom_args_dtype" "$transfer_custom_args_vals"

unique_id="${unique_id:-$(date '+%Y_%m_%d__%H_%M_%S')}"
bundle_name="$(basename "$bundle_dir")"
bundle_token="$(sanitize_token "$bundle_name")"
run_label="${RNAME:-${RUN_NAME:-BundleRun}}"
suffix="${RNAME_SUFFIX:-Transfer}"

export SOURCE_BUNDLE_PATH="$bundle_dir"
export SOURCE_BUNDLE_FILE="$bundle_file"
export SOURCE_RUN_METADATA_DIR="$source_run_metadata_dir"
export SOURCE_RESOLVED_CONFIG_PATH="$source_resolved_config"

export EVAL="${EVAL:-1}"
export RESUME="${RESUME:-0}"
export MPATH="$bundle_dir"
export MNAME="$checkpoint_file"
export SHM_NS="${SHM_NS_EVAL_BASE:-eval_${bundle_token}}${unique_id}"
export RNAME="${run_label}_${suffix}"
export COMMENT="${COMMENT_PREFIX:-bundle_transfer} from ${bundle_name}"

for cfg_override in "${cfg_overrides[@]}"; do
    cfg_override_name="${cfg_override%%=*}"
    if [[ "$cfg_override" != *=* || ! "$cfg_override_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid --set override: $cfg_override"
        exit 1
    fi
    if (( ! allow_contract_override )) && is_contract_var "$cfg_override_name"; then
        echo "launch_bundle.sh: refusing bundle contract override '$cfg_override_name'."
        echo "launch_bundle.sh: pass --allow_contract_override if this mismatch is intentional."
        exit 1
    fi
    export "$cfg_override"
done

world_iface_fname="${WORLD_INTERFACE:-}"
world_headless="${WORLD_HEADLESS:-1}"
world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
world_use_gpu="${USE_GPU_SIM:-1}"
world_env_profile="isaac5x"
world_jnt_imp_config_path="${JNT_IMP_CONFIG_PATH:-}"

case "$world_iface_fname" in
    *xmj_world_interface*)
        if [ "$N_ENVS" != "1" ]; then
            echo "launch_bundle.sh: XMJ transfer supports only N_ENVS=1."
            exit 1
        fi
        prepare_resolved_xbot_runtime_config
        resolve_xmj_files_dir
        if [ -z "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
            echo "launch_bundle.sh: could not resolve an XBot2 config. Set XBOT_CONFIG or XBOT_CONFIG_PATH."
            exit 1
        fi
        if [ -z "${RESOLVED_XMJ_FILES_DIR_PATH:-}" ]; then
            echo "launch_bundle.sh: could not resolve XMJ files dir. Set XMJ_FILES_DIR or XMJ_FILES_DIR_PATH."
            exit 1
        fi
        export XBOT_CONFIG_PATH="$RESOLVED_XBOT_CONFIG_PATH"
        export XMJ_FILES_DIR_PATH="$RESOLVED_XMJ_FILES_DIR_PATH"
        export USE_GPU_SIM="0"
        world_headless="${XMJ_HEADLESS:-0}"
        world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-0}"
        world_use_gpu=0
        world_env_profile="xbot"
        append_custom_arg "xmj_files_dir" "str" "$RESOLVED_XMJ_FILES_DIR_PATH"
        append_custom_arg "xmj_timeout" "int" "${XMJ_TIMEOUT_MS:-30000}"
        append_custom_arg "add_remote_exit_flag" "bool" "${ADD_REMOTE_EXIT_FLAG:-true}"
        ;;
    *rt_deploy_world_interface*)
        if [ "$N_ENVS" != "1" ]; then
            echo "launch_bundle.sh: RT transfer supports only N_ENVS=1."
            exit 1
        fi
        prepare_resolved_xbot_runtime_config
        if [ -z "${RESOLVED_XBOT_CONFIG_PATH:-}" ]; then
            echo "launch_bundle.sh: could not resolve an XBot2 config. Set XBOT_CONFIG or XBOT_CONFIG_PATH."
            exit 1
        fi
        export XBOT_CONFIG_PATH="$RESOLVED_XBOT_CONFIG_PATH"
        export RT_XBOT_CONFIG_PATH="$RESOLVED_XBOT_CONFIG_PATH"
        export USE_GPU_SIM="0"
        world_headless="${RT_HEADLESS:-0}"
        world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-0}"
        world_use_gpu=0
        world_env_profile="xbot"
        append_custom_arg "time_source" "str" "${TIME_SOURCE:-sim}"
        append_custom_arg "add_remote_exit_flag" "bool" "${ADD_REMOTE_EXIT_FLAG:-true}"
        ;;
    *isaac5x_world_interface*)
        world_headless="${WORLD_HEADLESS:-1}"
        world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
        world_use_gpu="${USE_GPU_SIM:-1}"
        world_env_profile="isaac5x"
        ;;
    *genesis_world_interface*)
        world_headless="${GENESIS_HEADLESS:-0}"
        world_use_custom_jnt_imp="${WORLD_USE_CUSTOM_JNT_IMP:-1}"
        world_use_gpu="${USE_GPU_SIM:-1}"
        world_env_profile="genesis"
        ;;
    *)
        echo "launch_bundle.sh: unsupported WORLD_INTERFACE: $world_iface_fname"
        exit 1
        ;;
esac

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
cp "$config_file" "${base_log_dir}/"

log_world="${base_log_dir}/ibrido_world_interface_${run_label}_${unique_id}.log"
log_cluster="${base_log_dir}/ibrido_mpc_cluster_${run_label}_${unique_id}.log"
log_eval="${base_log_dir}/ibrido_eval_env_${run_label}_${unique_id}.log"

echo "
launch_bundle.sh: logging output to->
source bundle: $bundle_file
transfer cfg: $config_file
remote env: $log_world
rhc cluster: $log_cluster
eval env: $log_eval
"

if (( ${SET_ULIM:-0} )); then
    ulimit -n "${ULIM_N:-131072}"
fi

echo "Will use shared memory namespace ${SHM_NS}"

export IBRIDO_RUN_META_DIR="${base_log_dir}/metadata"

ibrido_build_world_cmd "$world_iface_fname" "$N_ENVS" "$world_headless" "$world_use_custom_jnt_imp" "$world_use_gpu" "$world_jnt_imp_config_path"
remote_env_cmd="$IBRIDO_WORLD_CMD"

ibrido_build_cluster_cmd "$N_ENVS" 1 0
cluster_cmd="$IBRIDO_CLUSTER_CMD"

ibrido_build_training_cmd
eval_env_cmd="$IBRIDO_TRAINING_CMD"

world_launch_cmd="python $LRHC_DIR/launch_world_interface.py $remote_env_cmd"
cluster_launch_cmd="python $LRHC_DIR/launch_control_cluster.py $cluster_cmd"
eval_launch_cmd="python $LRHC_DIR/launch_train_env.py $eval_env_cmd --comment \"\\\"$COMMENT\\\"\""

ibrido_prepare_run_metadata "$config_file" "$world_launch_cmd" "$cluster_launch_cmd" "$eval_launch_cmd" "$unique_id"

if (( dry_run )); then
    ibrido_print_dry_run "$world_launch_cmd" "$cluster_launch_cmd" "$eval_launch_cmd"
    trap - EXIT INT TERM
    exit 0
fi

if [ "$world_env_profile" = "isaac5x" ]; then
    (
        micromamba activate "${MAMBA_ENV_NAME_ISAAC:-ibrido_isaac_py11}"
        source /isaac-sim/setup_conda_env.sh || exit 1
        source "$HOME/ibrido_ws/setup.bash" || exit 1
        export EXP_PATH="/isaac-sim/apps"
        exec python "$LRHC_DIR/launch_world_interface.py" $remote_env_cmd
    ) > "$log_world" 2>&1 &
else
    (
        micromamba activate "${MAMBA_ENV_NAME:-ibrido}"
        source "$HOME/ibrido_ws/setup.bash" || exit 1
        exec python "$LRHC_DIR/launch_world_interface.py" $remote_env_cmd
    ) > "$log_world" 2>&1 &
fi
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
primary_pid=$eval_env_pid
child_pids+=("$eval_env_pid")

wait_for_pid "$eval_env_pid"
eval_env_status=$?
cleanup
trap - EXIT INT TERM
exit "$eval_env_status"
