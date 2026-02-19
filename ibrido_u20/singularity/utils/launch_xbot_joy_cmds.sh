#!/bin/bash
# Launch joystick/keyboard agent commands with a provided shared memory namespace (IsaacSim stack).

usage() {
  echo "Usage: $0 --ns <shm_namespace> [--agent_refs] [--mode <linvel|pos>]"
  exit 1
}

NS=""
MODE="linvel"
AGENT_REFS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns) NS="$2"; shift ;;
    --mode) MODE="$2"; shift ;;
    --agent_refs) AGENT_REFS=1 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
  shift
done

if [[ -z "$NS" ]]; then
  echo "Error: --ns is required"
  usage
fi

if [[ "$MODE" != "linvel" && "$MODE" != "pos" ]]; then
  echo "Error: --mode must be one of: linvel, pos"
  usage
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME}"

cd "$HOME/ibrido_ws/src/AugMPCEnvs/aug_mpc_envs/scripts" || exit 1

CMD=(python launch_xbot2_joy_cmds.py --ns "$NS" --mode "$MODE")
if [[ "$AGENT_REFS" -eq 1 ]]; then
  CMD+=(--agent_refs)
fi

"${CMD[@]}"
