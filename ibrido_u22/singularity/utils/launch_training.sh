#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"

usage() {
  echo "Usage: $0 --robot_pkg_name PKG_NAME --robot_pkg_pref_path PKG_PREF_PATH --cocluster_dir COCLUSTER_DIR [--num_envs NUM] [--set_ulim|-ulim] [--ulim_n ULIM_N] \
    [--ns] [--run_name RUN_NAME] [--comment COMMENT] [--seed SEED] [--timeout_ms TIMEOUT] \
    [--codegen_override CG_OVERRIDE]"
  exit 1
}
num_envs=128
timeout_ms=60000
set_ulim=false
ulim_n=28672
seed=0
ns="ibrido"
run_name=""
comment=""
codegen_override=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --robot_pkg_name) robot_pkg_name="$2"; shift ;;
    --robot_pkg_pref_path) robot_pkg_pref_path="$2"; shift ;;
    --cocluster_dir) cocluster_dir="$2"; shift ;;
    --num_envs) num_envs="$2"; shift ;;
    --timeout_ms) timeout_ms="$2"; shift ;;
    -ulim|--set_ulim) set_ulim=true ;;
    --ulim_n) ulim_n="$2"; shift ;;
    --ns) ns="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --run_name) run_name="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    --codegen_override) codegen_override="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -z "$robot_pkg_name" ]; then
  echo "Error: --robot_pkg_name is mandatory."
  usage
fi

if [ -z "$robot_pkg_pref_path" ]; then
  echo "Error: --robot_pkg_pref_path is mandatory."
  usage
fi

if [ -z "$cocluster_dir" ]; then
  echo "Error: --cocluster_dir is mandatory."
  usage
fi

# activate micromamba for this shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${MAMBA_ENV_NAME}

wandb login --relogin $WANDB_KEY # login to wandb

source /isaac-sim/setup_conda_env.sh
source $HOME/ibrido_ws/setup.bash

if $set_ulim; then
  ulimit -n $ulim_n
fi

robot_pkg_pref_path_eval=$(eval echo $robot_pkg_pref_path)
codegen_override_eval=$(eval echo $codegen_override)
cocluster_dir_eval=$(eval echo $cocluster_dir)

python $LRHC_DIR/launch_sim_env.py --headless --remote_stepping --robot_name $ns --robot_pkg_name $robot_pkg_name --robot_pkg_pref_path $robot_pkg_pref_path_eval --num_envs $num_envs --timeout_ms $timeout_ms&
python $cocluster_dir_eval/launch_control_cluster.py --ns $ns --size $num_envs --timeout_ms $timeout_ms --codegen_override_dir $codegen_override_eval --robot_pkg_pref_path $robot_pkg_pref_path_eval & 
python $LRHC_DIR/launch_train_env.py --ns $ns --run_name $run_name --drop_dir $HOME/training_data --dump_checkpoints --comment $comment --seed $seed --timeout_ms $timeout_ms&

wait # wait for all to exit