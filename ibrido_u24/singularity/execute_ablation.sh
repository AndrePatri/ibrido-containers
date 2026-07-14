#!/bin/bash

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been set. Please set it to \${path_to_ibrido-containers}/ibrido_u24/singularity."
    exit 1
fi

usage() {
    echo "Usage: $0 --cfg_dir <training_cfgs_subdir> [--recursive] [--dry-run]"
    exit 1
}

script_dir=$(dirname "$0")
cfg_rel=""
recursive=0
dry_run=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --cfg_dir) cfg_rel="$2"; shift ;;
        --recursive) recursive=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$cfg_rel" ]; then
    usage
fi

if [[ "$cfg_rel" = /* ]]; then
    cfg_dir="$cfg_rel"
    cfg_prefix="${cfg_rel#${script_dir}/files/training_cfgs/}"
else
    cfg_dir="$script_dir/files/training_cfgs/$cfg_rel"
    cfg_prefix="$cfg_rel"
fi

if [ ! -d "$cfg_dir" ]; then
    echo "The directory $cfg_dir is not valid. Please provide a valid directory."
    exit 1
fi

if (( recursive )); then
    mapfile -t cfg_files < <(find "$cfg_dir" -type f -name "*.yaml" | sort)
else
    mapfile -t cfg_files < <(find "$cfg_dir" -maxdepth 1 -type f -name "*.yaml" | sort)
fi

if [ ${#cfg_files[@]} -eq 0 ]; then
    echo "No YAML configuration files found in $cfg_dir."
    exit 1
fi

echo "Will run an ablation study with the configuration files:"
for file in "${cfg_files[@]}"; do
    echo "${file#${cfg_dir}/}"
done

for file in "${cfg_files[@]}"; do
    # Stop sentinel, set by franklin/slurm/prescia_script.sh when the scheduler walltime is about to
    # expire. The runs are sequential, so signalling the one in flight is not enough: without this
    # guard the loop would start another training with minutes left on the clock, and that one would
    # be hard-killed with no model and no debug dump. Unset outside a scheduler -> no-op.
    if [ -n "${IBRIDO_STOP_FILE:-}" ] && [ -f "$IBRIDO_STOP_FILE" ]; then
        echo "execute_ablation.sh: stop requested (walltime approaching); skipping the remaining configs."
        break
    fi
    rel="${file#${cfg_dir}/}"
    cmd=("$script_dir/execute.sh" --cfg "${cfg_prefix}/${rel}")
    if (( dry_run )); then
        cmd+=(--dry-run)
    fi
    "${cmd[@]}"
done
echo "Ablation study completed."
