#!/bin/bash

IBRIDO_COMMAND_BUILDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ibrido_enabled() {
    case "${1:-0}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

ibrido_fail() {
    echo "ibrido: $*" >&2
    return 1
}

ibrido_require_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        ibrido_fail "required config variable ${name} is not set"
        return 1
    fi
}

ibrido_ws_src_path() {
    local path="$1"
    if [ -z "$path" ]; then
        return 1
    fi
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "${IBRIDO_WS_SRC:-${HOME}/ibrido_ws/src}" "$path"
    fi
}

ibrido_prepare_xbot_runtime_config() {
    local xbot_config="${1:-${XBOT_CONFIG_PATH:-${XBOT_CONFIG:-}}}"
    local impedance_config="${2:-${JNT_IMP_CONFIG_PATH:-}}"
    local token
    local output_dir
    local xbot_config_abs
    local impedance_config_abs
    local urdf_path_abs=""
    local srdf_path_abs=""
    local xrdf_output_dir=""
    local xrdf_paths=()
    local model_args=()

    if [ -z "$xbot_config" ]; then
        ibrido_fail "cannot prepare XBot runtime config: XBOT_CONFIG is not set"
        return 1
    fi
    if [ -z "$impedance_config" ]; then
        ibrido_fail "cannot prepare XBot runtime config: JNT_IMP_CONFIG_PATH is not set"
        return 1
    fi

    xbot_config_abs="$(ibrido_ws_src_path "$xbot_config")" || return 1
    impedance_config_abs="$(ibrido_ws_src_path "$impedance_config")" || return 1

    token="${SHM_NS:-${ROBOT_VARIANT:-xbot}}"
    token="${token//[^A-Za-z0-9_.-]/_}"
    output_dir="${IBRIDO_XBOT_RUNTIME_DIR:-/tmp/ibrido_xbot_configs/${token}}"

    if [ -n "${URDF_PATH:-}" ] || [ -n "${SRDF_PATH:-}" ]; then
        if [ -z "${URDF_PATH:-}" ] || [ -z "${SRDF_PATH:-}" ]; then
            ibrido_fail "cannot prepare XBot model paths: URDF_PATH and SRDF_PATH must both be set"
            return 1
        fi
        urdf_path_abs="$(ibrido_ws_src_path "$URDF_PATH")" || return 1
        srdf_path_abs="$(ibrido_ws_src_path "$SRDF_PATH")" || return 1
        xrdf_output_dir="${output_dir}/model"
        mapfile -t xrdf_paths < <(
            python3 "${IBRIDO_COMMAND_BUILDER_DIR}/ibrido_xrdf_builder.py" \
                --urdf-path "$urdf_path_abs" \
                --srdf-path "$srdf_path_abs" \
                --output-dir "$xrdf_output_dir" \
                --robot-name "$token" \
                --custom-args-names "${CUSTOM_ARGS_NAMES:-}" \
                --custom-args-dtype "${CUSTOM_ARGS_DTYPE:-}" \
                --custom-args-vals "${CUSTOM_ARGS_VALS:-}"
        ) || return 1
        if [ "${#xrdf_paths[@]}" -ne 2 ]; then
            ibrido_fail "XRDF builder did not return URDF and SRDF paths"
            return 1
        fi
        model_args+=(--urdf-path "${xrdf_paths[0]}" --srdf-path "${xrdf_paths[1]}")
    fi

    IBRIDO_XBOT_TEMPLATE_CONFIG_PATH="$xbot_config_abs"
    IBRIDO_XBOT_RUNTIME_CONFIG_PATH="$(
        python3 "${IBRIDO_COMMAND_BUILDER_DIR}/ibrido_xbot_config_builder.py" \
            --xbot-config "$xbot_config_abs" \
            --impedance-config "$impedance_config_abs" \
            "${model_args[@]}" \
            --output-dir "$output_dir"
    )" || return 1

    XBOT_CONFIG_PATH="$IBRIDO_XBOT_RUNTIME_CONFIG_PATH"
    RT_XBOT_CONFIG_PATH="$IBRIDO_XBOT_RUNTIME_CONFIG_PATH"
    if [ "${#xrdf_paths[@]}" -eq 2 ]; then
        IBRIDO_XBOT_RUNTIME_URDF_PATH="${xrdf_paths[0]}"
        IBRIDO_XBOT_RUNTIME_SRDF_PATH="${xrdf_paths[1]}"
        export IBRIDO_XBOT_RUNTIME_URDF_PATH IBRIDO_XBOT_RUNTIME_SRDF_PATH
    fi
    export IBRIDO_XBOT_TEMPLATE_CONFIG_PATH IBRIDO_XBOT_RUNTIME_CONFIG_PATH
    export XBOT_CONFIG_PATH RT_XBOT_CONFIG_PATH
}

ibrido_write_executable() {
    local path="$1"
    local cmd="$2"

    {
        printf '#!/bin/bash\n'
        printf 'set -e\n'
        printf '%s\n' "$cmd"
    } > "$path"
    chmod +x "$path"
}

ibrido_record_launch_command() {
    local name="$1"
    local cmd="$2"
    local metadata_dir="${3:-${IBRIDO_RUN_META_DIR:-}}"

    if [ -z "$metadata_dir" ]; then
        echo "ibrido_record_launch_command: metadata dir is not set"
        return 1
    fi
    if [[ ! "$name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ibrido_record_launch_command: invalid command name: $name"
        return 1
    fi

    mkdir -p "${metadata_dir}/launch" || return 1
    ibrido_write_executable "${metadata_dir}/launch/${name}.sh" "$cmd"
}

ibrido_write_launch_index() {
    local metadata_dir="${1:-${IBRIDO_RUN_META_DIR:-}}"
    local script
    local name

    if [ -z "$metadata_dir" ]; then
        echo "ibrido_write_launch_index: metadata dir is not set"
        return 1
    fi

    {
        printf 'schema: ibrido_u24_launch_commands_v1\n'
        printf 'commands:\n'
        for script in "${metadata_dir}/launch/"*.sh; do
            [ -f "$script" ] || continue
            name="$(basename "$script" .sh)"
            printf '  %s: ' "$name"
            ibrido_yaml_quote "launch/${name}.sh"
            printf '\n'
        done
    } > "${metadata_dir}/launch/commands.yaml"
}

ibrido_yaml_quote() {
    local value="${1:-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    printf '"%s"' "$value"
}

ibrido_bool_literal() {
    if ibrido_enabled "${1:-0}"; then
        printf 'true'
    else
        printf 'false'
    fi
}

ibrido_yaml_scalar_for_dtype() {
    local dtype="$1"
    local value="$2"

    case "$dtype" in
        bool)
            ibrido_bool_literal "$value"
            ;;
        int|integer|float|double)
            if [[ "$value" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
                printf '%s' "$value"
            else
                ibrido_yaml_quote "$value"
            fi
            ;;
        *)
            ibrido_yaml_quote "$value"
            ;;
    esac
}

ibrido_assert_custom_args_triplets() {
    local names_s="${1:-${CUSTOM_ARGS_NAMES:-}}"
    local dtype_s="${2:-${CUSTOM_ARGS_DTYPE:-}}"
    local vals_s="${3:-${CUSTOM_ARGS_VALS:-}}"
    local names=()
    local dtypes=()
    local vals=()

    read -r -a names <<< "$names_s"
    read -r -a dtypes <<< "$dtype_s"
    read -r -a vals <<< "$vals_s"

    if [ "${#names[@]}" -ne "${#dtypes[@]}" ] || [ "${#names[@]}" -ne "${#vals[@]}" ]; then
        ibrido_fail "custom args triplet length mismatch: names=${#names[@]}, dtype=${#dtypes[@]}, vals=${#vals[@]}"
        return 1
    fi
}

ibrido_write_custom_args_mapping() {
    local indent="$1"
    local names_s="${2:-${CUSTOM_ARGS_NAMES:-}}"
    local dtype_s="${3:-${CUSTOM_ARGS_DTYPE:-}}"
    local vals_s="${4:-${CUSTOM_ARGS_VALS:-}}"
    local names=()
    local dtypes=()
    local vals=()
    local i
    local key

    read -r -a names <<< "$names_s"
    read -r -a dtypes <<< "$dtype_s"
    read -r -a vals <<< "$vals_s"

    if [ "${#names[@]}" -eq 0 ]; then
        printf '%s{}\n' "$indent"
        return 0
    fi

    for i in "${!names[@]}"; do
        key="${names[$i]}"
        if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            printf '%s%s:\n' "$indent" "$key"
        else
            printf '%s' "$indent"
            ibrido_yaml_quote "$key"
            printf ':\n'
        fi
        printf '%s  dtype: ' "$indent"
        ibrido_yaml_quote "${dtypes[$i]}"
        printf '\n'
        printf '%s  value: ' "$indent"
        ibrido_yaml_scalar_for_dtype "${dtypes[$i]}" "${vals[$i]}"
        printf '\n'
    done
}

ibrido_resolve_time_source() {
    TIME_SOURCE="${TIME_SOURCE:-sim}"
    export TIME_SOURCE
}

ibrido_normalize_runtime_config() {
    if [ -n "${JNT_IMP_CF_PATH:-}" ]; then
        ibrido_fail "JNT_IMP_CF_PATH is deprecated in u24; use JNT_IMP_CONFIG_PATH"
        return 1
    fi

    # NOTE: the world interfaces (XMJ / RT deploy) now generate URDF/SRDF and the
    # runtime XBot config themselves from the template xbot config + template
    # impedance config (see ibrido_build_world_cmd / _prepare_xbot_runtime_config),
    # so we no longer pre-generate a runtime XBot config here. XBOT_CONFIG_PATH stays
    # the template; the external xbot2-core terminal (byobu RT sim) still prepares a
    # runtime config via prepare_resolved_xbot_runtime_config.
    XBOT_CONFIG_PATH="${XBOT_CONFIG_PATH:-${XBOT_CONFIG:-}}"
    RT_XBOT_CONFIG_PATH="${RT_XBOT_CONFIG_PATH:-${XBOT_CONFIG_PATH:-}}"
    SITE_PROFILE="${SITE_PROFILE:-local_dev}"
    ZMQ_BRIDGE_BIND_IP="${ZMQ_BRIDGE_BIND_IP:-0.0.0.0}"
    ZMQ_BRIDGE_SOURCE_IP="${ZMQ_BRIDGE_SOURCE_IP:-127.0.0.1}"
    ZMQ_BRIDGE_PORT_BASE="${ZMQ_BRIDGE_PORT_BASE:-20000}"
    ZMQ_BRIDGE_PORT_SPAN="${ZMQ_BRIDGE_PORT_SPAN:-40000}"

    export XBOT_CONFIG_PATH RT_XBOT_CONFIG_PATH SITE_PROFILE
    export ZMQ_BRIDGE_BIND_IP ZMQ_BRIDGE_SOURCE_IP ZMQ_BRIDGE_PORT_BASE ZMQ_BRIDGE_PORT_SPAN

    ibrido_resolve_time_source
    ibrido_assert_custom_args_triplets
}

ibrido_validate_runtime_config() {
    local required_vars=(
        SHM_NS
        N_ENVS
        URDF_PATH
        SRDF_PATH
        JNT_IMP_CONFIG_PATH
        CLUSTER_CL_FNAME
        CLUSTER_DT
        PHYSICS_DT
        N_NODES
        TRAIN_ENV_FNAME
        TRAIN_ENV_CNAME
        SEED
        TIMEOUT_MS
        CUSTOM_ARGS_NAMES
        CUSTOM_ARGS_DTYPE
        CUSTOM_ARGS_VALS
    )
    local name

    for name in "${required_vars[@]}"; do
        ibrido_require_var "$name" || return 1
    done
}

ibrido_write_filtered_env() {
    local path="$1"

    export -p | grep -Ev         '(^declare -x (WANDB_KEY|NGC_KEY|ISAAC_KEY|API_KEY|TOKEN|SECRET|PASSWORD)=|KEY=|TOKEN=|SECRET=|PASSWORD=|^declare -x (HOME|PWD|OLDPWD|PATH|LD_LIBRARY_PATH|PYTHONPATH|SHELL|USER|USERNAME|LOGNAME)=|^declare -x (APPTAINER_|SINGULARITY_|BYOBU_|VSCODE_|GNOME_|GDM|XDG_|DBUS_|SSH_AUTH_SOCK|DISPLAY|XAUTHORITY|WINDOWPATH|SESSION_MANAGER))'         > "$path"
}

ibrido_snapshot_config_stack() {
    local metadata_dir="$1"
    local config_file="$2"
    local cfg_root="${IBRIDO_CFG_BASEPATH:-${HOME}/ibrido_files/training_cfgs}"
    local stack="${IBRIDO_CFG_STACK:-}"
    local entry
    local rel
    local snapshot_rel
    local first_snapshot=""
    local old_ifs

    mkdir -p "${metadata_dir}/configs" || return 1
    : > "${metadata_dir}/cfg_stack.txt"
    {
        printf 'schema: ibrido_u24_config_stack_v1\n'
        printf 'configs:\n'
    } > "${metadata_dir}/configs/index.yaml"

    if [ -n "$config_file" ]; then
        if [ -z "$stack" ]; then
            stack="$config_file"
        elif [[ ":$stack:" != *":$config_file:"* ]]; then
            stack="$config_file:$stack"
        fi
    fi

    old_ifs="$IFS"
    IFS=':'
    for entry in $stack; do
        IFS="$old_ifs"
        [ -n "$entry" ] || continue
        printf '%s\n' "$entry" >> "${metadata_dir}/cfg_stack.txt"
        if [ ! -f "$entry" ]; then
            {
                printf '  - original: '
                ibrido_yaml_quote "$entry"
                printf '\n'
                printf '    missing: true\n'
            } >> "${metadata_dir}/configs/index.yaml"
            IFS=':'
            continue
        fi

        if [[ "$entry" == "$cfg_root"/* ]]; then
            rel="${entry#${cfg_root}/}"
        else
            rel="external/$(basename "$entry")"
        fi
        snapshot_rel="configs/${rel}"
        mkdir -p "${metadata_dir}/$(dirname "$snapshot_rel")" || return 1
        cp "$entry" "${metadata_dir}/${snapshot_rel}" || return 1
        if [ -z "$first_snapshot" ]; then
            first_snapshot="$snapshot_rel"
        fi
        {
            printf '  - original: '
            ibrido_yaml_quote "$entry"
            printf '\n'
            printf '    snapshot: '
            ibrido_yaml_quote "$snapshot_rel"
            printf '\n'
        } >> "${metadata_dir}/configs/index.yaml"
        IFS=':'
    done
    IFS="$old_ifs"

    IBRIDO_CONFIG_SNAPSHOT_REL="$first_snapshot"
    export IBRIDO_CONFIG_SNAPSHOT_REL
}

ibrido_write_git_state() {
    local metadata_dir="$1"
    local workspace_src="${2:-${IBRIDO_WS_SRC:-${HOME}/ibrido_ws/src}}"
    local repo
    local name
    local commit
    local branch
    local remote
    local dirty
    local patch_rel
    local status_rel

    mkdir -p "${metadata_dir}/git/dirty_patches" "${metadata_dir}/git/dirty_status" || return 1

    {
        printf 'schema: ibrido_u24_git_state_v1\n'
        printf 'workspace_src: '
        ibrido_yaml_quote "$workspace_src"
        printf '\n'
        printf 'repos:\n'
    } > "${metadata_dir}/git/repos.yaml"

    if [ ! -d "$workspace_src" ]; then
        printf 'warnings:\n  - ' >> "${metadata_dir}/git/repos.yaml"
        ibrido_yaml_quote "workspace src not found: $workspace_src" >> "${metadata_dir}/git/repos.yaml"
        printf '\n' >> "${metadata_dir}/git/repos.yaml"
        return 0
    fi

    for repo in "${workspace_src}"/*; do
        [ -d "$repo/.git" ] || continue
        name="$(basename "$repo")"
        commit="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
        branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        remote="$(git -C "$repo" remote get-url origin 2>/dev/null || true)"
        dirty=0
        if [ -n "$(git -C "$repo" status --porcelain 2>/dev/null)" ]; then
            dirty=1
        fi

        patch_rel=""
        status_rel=""
        if ibrido_enabled "$dirty"; then
            status_rel="git/dirty_status/${name}.status"
            git -C "$repo" status --porcelain > "${metadata_dir}/${status_rel}" 2>/dev/null || true
            if git -C "$repo" diff --binary --quiet --exit-code 2>/dev/null; then
                patch_rel=""
            else
                patch_rel="git/dirty_patches/${name}.patch"
                git -C "$repo" diff --binary > "${metadata_dir}/${patch_rel}" 2>/dev/null || true
                [ -s "${metadata_dir}/${patch_rel}" ] || patch_rel=""
            fi
        fi

        {
            printf '  %s:\n' "$name"
            printf '    commit: '
            ibrido_yaml_quote "$commit"
            printf '\n'
            printf '    branch: '
            ibrido_yaml_quote "$branch"
            printf '\n'
            printf '    remote: '
            ibrido_yaml_quote "$remote"
            printf '\n'
            printf '    dirty: %s\n' "$(ibrido_bool_literal "$dirty")"
            if [ -n "$status_rel" ]; then
                printf '    status: '
                ibrido_yaml_quote "$status_rel"
                printf '\n'
            fi
            if [ -n "$patch_rel" ]; then
                printf '    patch: '
                ibrido_yaml_quote "$patch_rel"
                printf '\n'
            fi
        } >> "${metadata_dir}/git/repos.yaml"
    done
}

ibrido_build_world_cmd() {
    local world_iface_fname="$1"
    local num_envs="$2"
    local headless="$3"
    local use_custom_jnt_imp="$4"
    local use_gpu="$5"
    local jnt_imp_config_path="${6:-${JNT_IMP_CONFIG_PATH:-}}"
    local custom_args_names="${7:-${CUSTOM_ARGS_NAMES:-}}"
    local custom_args_dtype="${8:-${CUSTOM_ARGS_DTYPE:-}}"
    local custom_args_vals="${9:-${CUSTOM_ARGS_VALS:-}}"

    # XBot-backed interfaces (XMJ / RT deploy) own their runtime model+config: the
    # world interface generates URDF/SRDF and builds the runtime XBot config from the
    # *template* xbot config + the *template* impedance config. So pass the template
    # impedance via --jnt_imp_config_path and the template xbot config via the
    # xbot_config_path custom arg, instead of a launcher-prepared runtime config.
    case "$world_iface_fname" in
        *xmj_world_interface*|*rt_deploy_world_interface*)
            jnt_imp_config_path="${JNT_IMP_CONFIG_PATH:-$jnt_imp_config_path}"
            local xbot_config_template="${IBRIDO_XBOT_TEMPLATE_CONFIG_PATH:-${RESOLVED_XBOT_CONFIG:-${XBOT_CONFIG:-}}}"
            if [ -n "$xbot_config_template" ] && [[ " ${custom_args_names} " != *" xbot_config_path "* ]]; then
                local xbot_config_template_abs
                xbot_config_template_abs="$(ibrido_ws_src_path "$xbot_config_template")" || return 1
                custom_args_names+=" xbot_config_path"
                custom_args_dtype+=" str"
                custom_args_vals+=" ${xbot_config_template_abs}"
            fi
            ;;
    esac

    IBRIDO_WORLD_INTERFACE="$world_iface_fname"
    IBRIDO_WORLD_NUM_ENVS="$num_envs"
    IBRIDO_WORLD_HEADLESS="$headless"
    IBRIDO_WORLD_USE_GPU="$use_gpu"

    IBRIDO_WORLD_CMD="--robot_name $SHM_NS \
--urdf_path $URDF_PATH --srdf_path  $SRDF_PATH \
--jnt_imp_config_path $jnt_imp_config_path \
--cluster_dt $CLUSTER_DT \
--physics_dt $PHYSICS_DT \
--n_contacts ${N_CONTACTS:-4} \
--num_envs $num_envs --seed $SEED --timeout_ms $TIMEOUT_MS \
--world_iface_fname $world_iface_fname \
--custom_args_names $custom_args_names \
--custom_args_dtype $custom_args_dtype \
--custom_args_vals $custom_args_vals "

    if ibrido_enabled "$headless"; then
        IBRIDO_WORLD_CMD="--headless $IBRIDO_WORLD_CMD"
    fi
    if ibrido_enabled "$use_custom_jnt_imp"; then
        IBRIDO_WORLD_CMD+="--use_custom_jnt_imp "
    fi
    if ibrido_enabled "$REMOTE_STEPPING"; then
        IBRIDO_WORLD_CMD+="--remote_stepping "
    fi
    if ibrido_enabled "$use_gpu"; then
        IBRIDO_WORLD_CMD+="--use_gpu "
    fi
}

ibrido_build_cluster_cmd() {
    local cluster_size="$1"
    local include_codegen_override="${2:-0}"
    local include_debug="${3:-1}"

    IBRIDO_CLUSTER_CMD="--ns $SHM_NS --size $cluster_size --timeout_ms $TIMEOUT_MS "
    if ibrido_enabled "$include_codegen_override" && [ -n "${CODEGEN_OVERRIDE_BDIR:-}" ]; then
        IBRIDO_CLUSTER_CMD+="--codegen_override_dir $CODEGEN_OVERRIDE_BDIR "
    fi
    IBRIDO_CLUSTER_CMD+="\
--urdf_path $URDF_PATH --srdf_path $SRDF_PATH --cluster_client_fname $CLUSTER_CL_FNAME \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS \
--cluster_dt $CLUSTER_DT \
--n_nodes $N_NODES "

    if ibrido_enabled "$include_debug" && ibrido_enabled "$CLUSTER_DB"; then
        IBRIDO_CLUSTER_CMD+="--enable_debug "
    fi
    if ibrido_enabled "$IS_CLOSED_LOOP"; then
        IBRIDO_CLUSTER_CMD+="--cloop "
    fi
}

ibrido_build_training_cmd() {
    IBRIDO_TRAINING_CMD="--ns $SHM_NS --drop_dir $HOME/training_data \
--seed $SEED --timeout_ms $TIMEOUT_MS \
--reset_on_init \
--env_fname $TRAIN_ENV_FNAME --env_classname $TRAIN_ENV_CNAME \
--demo_stop_thresh $DEMO_STOP_THRESH  \
--actor_lwidth $ACTOR_LWIDTH --actor_n_hlayers $ACTOR_DEPTH \
--critic_lwidth $CRITIC_LWIDTH --critic_n_hlayers $CRITIC_DEPTH \
--tot_tsteps $TOT_STEPS \
--demo_envs_perc $DEMO_ENVS_PERC \
--expl_envs_perc $EXPL_ENVS_PERC \
--action_repeat $ACTION_REPEAT \
--compression_ratio $COMPRESSION_RATIO \
--discount_factor $DISCOUNT_FACTOR \
--custom_args_names $CUSTOM_ARGS_NAMES \
--custom_args_dtype $CUSTOM_ARGS_DTYPE \
--custom_args_vals $CUSTOM_ARGS_VALS "

    if ibrido_enabled "$DUMP_ENV_CHECKPOINTS"; then
        IBRIDO_TRAINING_CMD+="--dump_checkpoints "
    fi
    if ibrido_enabled "${STEP_WHILE_SETUP:-0}"; then
        IBRIDO_TRAINING_CMD+="--step_while_setup "
    fi
    if ibrido_enabled "$USE_DUMMY"; then
        IBRIDO_TRAINING_CMD+="--dummy "
    elif ibrido_enabled "$USE_SAC"; then
        IBRIDO_TRAINING_CMD+="--sac "
    fi
    if ibrido_enabled "$DEBUG"; then
        IBRIDO_TRAINING_CMD+="--db --env_db "
    fi
    if ibrido_enabled "$RMDEBUG"; then
        IBRIDO_TRAINING_CMD+="--rmdb "
    fi
    if ibrido_enabled "$DUMP_ENV_CHECKPOINTS" && ibrido_enabled "$DEBUG"; then
        IBRIDO_TRAINING_CMD+="--full_env_db "
    fi
    if ibrido_enabled "$USE_RND"; then
        IBRIDO_TRAINING_CMD+="--use_rnd "
    fi
    if ibrido_enabled "$OBS_NORM"; then
        IBRIDO_TRAINING_CMD+="--obs_norm "
    fi
    if ibrido_enabled "$OBS_RESCALING"; then
        IBRIDO_TRAINING_CMD+="--obs_rescale "
    fi
    if ibrido_enabled "$WEIGHT_NORM"; then
        IBRIDO_TRAINING_CMD+="--add_weight_norm "
    fi
    if ibrido_enabled "$LAYER_NORM"; then
        IBRIDO_TRAINING_CMD+="--add_layer_norm "
    fi
    if ibrido_enabled "$BATCH_NORM"; then
        IBRIDO_TRAINING_CMD+="--add_batch_norm "
    fi
    if ibrido_enabled "$CRITIC_ACTION_RESCALE"; then
        IBRIDO_TRAINING_CMD+="--act_rescale_critic "
    fi
    if ibrido_enabled "$USE_PERIOD_RESETS"; then
        IBRIDO_TRAINING_CMD+="--use_period_resets "
    fi
    if [[ -n "${RNAME:-}" ]]; then
        IBRIDO_TRAINING_CMD+="--run_name ${RNAME}_${TRAIN_ENV_CNAME} "
    fi
    if ibrido_enabled "$RESUME"; then
        IBRIDO_TRAINING_CMD+="--resume --mpath $MPATH --mname $MNAME "
        if ibrido_enabled "$OVERRIDE_ENV"; then
            IBRIDO_TRAINING_CMD+="--override_env "
        fi
    fi
    if ibrido_enabled "$EVAL"; then
        IBRIDO_TRAINING_CMD+="--eval --n_eval_timesteps $TOT_STEPS --mpath $MPATH --mname $MNAME "
        if ibrido_enabled "$DET_EVAL"; then
            IBRIDO_TRAINING_CMD+="--det_eval "
        fi
        if ibrido_enabled "$EVAL_ON_CPU"; then
            IBRIDO_TRAINING_CMD+="--use_cpu "
        fi
        if ibrido_enabled "$OVERRIDE_ENV"; then
            IBRIDO_TRAINING_CMD+="--override_env "
        fi
        if ibrido_enabled "$OVERRIDE_AGENT_REFS"; then
            IBRIDO_TRAINING_CMD+="--override_agent_refs "
        fi
    fi
    if [ -n "${IBRIDO_RUN_META_DIR:-}" ]; then
        IBRIDO_TRAINING_CMD+="--run_meta_dir $IBRIDO_RUN_META_DIR "
    fi
}

ibrido_prepare_run_metadata() {
    local config_file="$1"
    local world_launch_cmd="$2"
    local cluster_launch_cmd="$3"
    local training_launch_cmd="$4"
    local run_id="${5:-${unique_id:-manual}}"
    local run_label="${RUN_NAME:-${RNAME:-IbridoRun}}"
    local jnt_imp_config_path
    local xbot_config_path
    local rt_xbot_config_path

    ibrido_normalize_runtime_config || return 1
    ibrido_validate_runtime_config || return 1

    jnt_imp_config_path="${JNT_IMP_CONFIG_PATH:-}"
    xbot_config_path="${XBOT_CONFIG_PATH:-}"
    rt_xbot_config_path="${RT_XBOT_CONFIG_PATH:-${xbot_config_path:-}}"

    export IBRIDO_RUN_META_DIR="${IBRIDO_RUN_META_DIR:-${HOME}/ibrido_logs/ibrido_run_${run_id}/metadata}"
    mkdir -p "${IBRIDO_RUN_META_DIR}/configs" "${IBRIDO_RUN_META_DIR}/launch" "${IBRIDO_RUN_META_DIR}/git" || return 1

    ibrido_snapshot_config_stack "$IBRIDO_RUN_META_DIR" "$config_file" || return 1
    local config_snapshot_rel="${IBRIDO_CONFIG_SNAPSHOT_REL:-}"

    {
        printf 'schema: ibrido_u24_run_manifest_v1\n'
        printf 'run_id: '
        ibrido_yaml_quote "$run_id"
        printf '\n'
        printf 'run_label: '
        ibrido_yaml_quote "$run_label"
        printf '\n'
        printf 'config_file: '
        ibrido_yaml_quote "$config_file"
        printf '\n'
        if [ -n "$config_snapshot_rel" ]; then
            printf 'config_snapshot: '
            ibrido_yaml_quote "$config_snapshot_rel"
            printf '\n'
        fi
        if [ -n "${SOURCE_BUNDLE_PATH:-}" ]; then
            printf 'source_bundle: '
            ibrido_yaml_quote "$SOURCE_BUNDLE_PATH"
            printf '\n'
        fi
        if [ -n "${SOURCE_BUNDLE_FILE:-}" ]; then
            printf 'source_bundle_file: '
            ibrido_yaml_quote "$SOURCE_BUNDLE_FILE"
            printf '\n'
        fi
        if [ -n "${SOURCE_RUN_METADATA_DIR:-}" ]; then
            printf 'source_run_metadata: '
            ibrido_yaml_quote "$SOURCE_RUN_METADATA_DIR"
            printf '\n'
        fi
        printf 'shm_namespace: '
        ibrido_yaml_quote "$SHM_NS"
        printf '\n'
        printf 'created_from: "ibrido_u24"\n'
        printf 'created_at: '
        ibrido_yaml_quote "$(date -Iseconds)"
        printf '\n'
        printf 'resolved_config: "resolved_config.yaml"\n'
        printf 'resolved_env: "resolved_env.sh"\n'
        printf 'cfg_stack: "cfg_stack.txt"\n'
        printf 'config_stack_index: "configs/index.yaml"\n'
        printf 'git_state: "git/repos.yaml"\n'
        printf 'launch_index: "launch/commands.yaml"\n'
        printf 'launch_commands:\n'
        printf '  world_interface: "launch/world_interface.sh"\n'
        printf '  control_cluster: "launch/control_cluster.sh"\n'
        printf '  train_env: "launch/train_env.sh"\n'
    } > "${IBRIDO_RUN_META_DIR}/run_manifest.yaml"

    {
        printf 'schema: ibrido_u24_resolved_config_v1\n'
        if [ -n "${SOURCE_BUNDLE_PATH:-}" ]; then
            printf 'source:\n'
            printf '  bundle_path: '
            ibrido_yaml_quote "$SOURCE_BUNDLE_PATH"
            printf '\n'
            printf '  bundle_file: '
            ibrido_yaml_quote "${SOURCE_BUNDLE_FILE:-}"
            printf '\n'
            printf '  run_metadata: '
            ibrido_yaml_quote "${SOURCE_RUN_METADATA_DIR:-}"
            printf '\n'
            printf '  resolved_config: '
            ibrido_yaml_quote "${SOURCE_RESOLVED_CONFIG_PATH:-}"
            printf '\n'
        fi
        printf 'robot:\n'
        printf '  name: '
        ibrido_yaml_quote "$SHM_NS"
        printf '\n'
        printf '  urdf_path: '
        ibrido_yaml_quote "$URDF_PATH"
        printf '\n'
        printf '  srdf_path: '
        ibrido_yaml_quote "$SRDF_PATH"
        printf '\n'
        printf '  n_contacts: %s\n' "${N_CONTACTS:-4}"
        printf '  family: '
        ibrido_yaml_quote "${ROBOT_FAMILY:-}"
        printf '\n'
        printf '  variant: '
        ibrido_yaml_quote "${ROBOT_VARIANT:-}"
        printf '\n'
        printf 'controller:\n'
        printf '  cluster_client: '
        ibrido_yaml_quote "$CLUSTER_CL_FNAME"
        printf '\n'
        printf '  cluster_dt: %s\n' "$CLUSTER_DT"
        printf '  physics_dt: %s\n' "$PHYSICS_DT"
        printf '  n_nodes: %s\n' "$N_NODES"
        printf '  closed_loop: %s\n' "$(ibrido_bool_literal "$IS_CLOSED_LOOP")"
        printf '  custom_args:\n'
        printf '    typed:\n'
        ibrido_write_custom_args_mapping '      ' "${CUSTOM_ARGS_NAMES:-}" "${CUSTOM_ARGS_DTYPE:-}" "${CUSTOM_ARGS_VALS:-}"
        printf '    raw:\n'
        printf '      names: '
        ibrido_yaml_quote "${CUSTOM_ARGS_NAMES:-}"
        printf '\n'
        printf '      dtype: '
        ibrido_yaml_quote "${CUSTOM_ARGS_DTYPE:-}"
        printf '\n'
        printf '      vals: '
        ibrido_yaml_quote "${CUSTOM_ARGS_VALS:-}"
        printf '\n'
        printf 'world:\n'
        printf '  interface: '
        ibrido_yaml_quote "${IBRIDO_WORLD_INTERFACE:-${WORLD_INTERFACE:-}}"
        printf '\n'
        printf '  num_envs: %s\n' "${IBRIDO_WORLD_NUM_ENVS:-$N_ENVS}"
        printf '  headless: %s\n' "$(ibrido_bool_literal "${IBRIDO_WORLD_HEADLESS:-0}")"
        printf '  use_gpu: %s\n' "$(ibrido_bool_literal "${IBRIDO_WORLD_USE_GPU:-${USE_GPU_SIM:-0}}")"
        printf '  remote_stepping: %s\n' "$(ibrido_bool_literal "$REMOTE_STEPPING")"
        printf 'time:\n'
        printf '  source: '
        ibrido_yaml_quote "${TIME_SOURCE:-sim}"
        printf '\n'
        printf '  realtime_factor: '
        ibrido_yaml_quote "${REALTIME_FACTOR:-1.0}"
        printf '\n'
        printf '  timeout_ms: %s\n' "$TIMEOUT_MS"
        printf 'xbot:\n'
        printf '  jnt_imp_config_path: '
        ibrido_yaml_quote "$jnt_imp_config_path"
        printf '\n'
        printf '  xbot_config_path: '
        ibrido_yaml_quote "$xbot_config_path"
        printf '\n'
        printf '  rt_xbot_config_path: '
        ibrido_yaml_quote "$rt_xbot_config_path"
        printf '\n'
        printf '  xmj_files_dir: '
        ibrido_yaml_quote "${XMJ_FILES_DIR:-}"
        printf '\n'
        printf '  xmj_simopt_path: '
        ibrido_yaml_quote "${XMJ_SIMOPT_PATH:-}"
        printf '\n'
        printf '  xmj_world_path: '
        ibrido_yaml_quote "${XMJ_WORLD_PATH:-}"
        printf '\n'
        printf '  xmj_sites_path: '
        ibrido_yaml_quote "${XMJ_SITES_PATH:-}"
        printf '\n'
        printf '  filter_profile: '
        ibrido_yaml_quote "${XBOT2_FILTER_PROFILE:-}"
        printf '\n'
        printf 'site:\n'
        printf '  name: '
        ibrido_yaml_quote "${SITE_PROFILE:-local_dev}"
        printf '\n'
        printf '  ros:\n'
        printf '    localhost_only: '
        ibrido_yaml_quote "${ROS_LOCALHOST_ONLY:-}"
        printf '\n'
        printf '    domain_id: '
        ibrido_yaml_quote "${ROS_DOMAIN_ID:-}"
        printf '\n'
        printf '  zmq:\n'
        printf '    bind_ip: '
        ibrido_yaml_quote "${ZMQ_BRIDGE_BIND_IP:-}"
        printf '\n'
        printf '    source_ip: '
        ibrido_yaml_quote "${ZMQ_BRIDGE_SOURCE_IP:-}"
        printf '\n'
        printf '    port_base: %s\n' "${ZMQ_BRIDGE_PORT_BASE:-20000}"
        printf '    port_span: %s\n' "${ZMQ_BRIDGE_PORT_SPAN:-40000}"
        printf '  resources:\n'
        printf '    zmq_bridge_cores: '
        ibrido_yaml_quote "${ZMQ_BRIDGE_CORES:-}"
        printf '\n'
        printf 'training:\n'
        printf '  algo: '
        if ibrido_enabled "${USE_DUMMY:-0}"; then
            ibrido_yaml_quote "dummy"
        elif ibrido_enabled "${USE_SAC:-0}"; then
            ibrido_yaml_quote "sac"
        else
            ibrido_yaml_quote "unknown"
        fi
        printf '\n'
        printf '  env_fname: '
        ibrido_yaml_quote "$TRAIN_ENV_FNAME"
        printf '\n'
        printf '  env_classname: '
        ibrido_yaml_quote "$TRAIN_ENV_CNAME"
        printf '\n'
        printf '  seed: %s\n' "$SEED"
        printf '  total_steps: %s\n' "$TOT_STEPS"
        printf '  action_repeat: %s\n' "$ACTION_REPEAT"
        printf '  obs_norm: %s\n' "$(ibrido_bool_literal "${OBS_NORM:-0}")"
        printf '  obs_rescale: %s\n' "$(ibrido_bool_literal "${OBS_RESCALING:-0}")"
        printf '  run_name: '
        ibrido_yaml_quote "${RNAME:-}"
        printf '\n'
        printf '  eval: %s\n' "$(ibrido_bool_literal "${EVAL:-0}")"
        printf '  resume: %s\n' "$(ibrido_bool_literal "${RESUME:-0}")"
        printf '  override_env: %s\n' "$(ibrido_bool_literal "${OVERRIDE_ENV:-0}")"
        printf '  mpath: '
        ibrido_yaml_quote "${MPATH:-}"
        printf '\n'
        printf '  mname: '
        ibrido_yaml_quote "${MNAME:-}"
        printf '\n'
    } > "${IBRIDO_RUN_META_DIR}/resolved_config.yaml"

    ibrido_write_filtered_env "${IBRIDO_RUN_META_DIR}/resolved_env.sh"
    ibrido_write_git_state "${IBRIDO_RUN_META_DIR}"

    ibrido_record_launch_command "world_interface" "$world_launch_cmd"
    ibrido_record_launch_command "control_cluster" "$cluster_launch_cmd"
    ibrido_record_launch_command "train_env" "$training_launch_cmd"
    ibrido_write_launch_index "${IBRIDO_RUN_META_DIR}"
}

ibrido_print_dry_run() {
    local world_launch_cmd="$1"
    local cluster_launch_cmd="$2"
    local training_launch_cmd="$3"

    printf 'IBRIDO_RUN_META_DIR=%s\n' "${IBRIDO_RUN_META_DIR:-}"
    printf '\n[world]\n%s\n' "$world_launch_cmd"
    printf '\n[cluster]\n%s\n' "$cluster_launch_cmd"
    printf '\n[training]\n%s\n' "$training_launch_cmd"
}
