#!/bin/bash
# Launch joystick/keyboard agent commands with a provided shared memory namespace (IsaacSim stack).

usage() {
  echo "Usage: $0 --ns <shm_namespace>"
  exit 1
}

NS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns) NS="$2"; shift ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
  shift
done

if [[ -z "$NS" ]]; then
  echo "Error: --ns is required"
  usage
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME}"

cd "$HOME/ibrido_ws/src/AugMPCEnvs/aug_mpc_envs/scripts" || exit 1

python utilities/launch_xbot2_joy_cmds.py \
    --ns "$NS"