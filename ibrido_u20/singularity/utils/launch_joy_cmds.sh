#!/bin/bash
# Launch joystick/keyboard agent commands with a provided shared memory namespace (IsaacSim stack).

usage() {
  echo "Usage: $0 --ns <shm_namespace> [--env_idx <idx>] [--rhc]"
  exit 1
}

NS=""
ENV_IDX=0
LAUNCH_RHC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns) NS="$2"; shift ;;
    --env_idx) ENV_IDX="$2"; shift ;;
    --rhc) LAUNCH_RHC=1 ;;
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

cd "$HOME/ibrido_ws/src/AugMPC/aug_mpc/scripts" || exit 1

if (( LAUNCH_RHC )); then
  python launch_rhc_keybrd_cmds.py \
    --ns "$NS" \
    --env_idx "$ENV_IDX" \
    --from_stdin \
    --add_remote_exit_flag \
    --joy
else
  python launch_agent_keybrd_cmds.py \
  --ns "$NS" \
  --env_idx "$ENV_IDX" \
  --agent_refs_world \
  --add_remote_exit_flag \
  --from_stdin \
  --joy
fi
