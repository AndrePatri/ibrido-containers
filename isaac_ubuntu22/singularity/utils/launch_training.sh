#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
DIR1="$WS_ROOT/src/LRHControl/lrhc_control/scripts"
DIR2="$WS_ROOT/src/KyonRLStepping/kyonrlstepping/scripts"

usage() {
  echo "Usage: $0 --robot_pkg_name PKG_NAME [--num_envs NUM] [--ulim_n ULIM_N] [--ns] [--run_name RUN_NAME] [--comment COMMENT]"
  exit 1
}
num_envs=128
ulim_n=28672
ns="ibrido"
run_name=""
comment=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --robot_pkg_name) robot_pkg_name="$2"; shift ;;
    --num_envs) num_envs="$2"; shift ;;
    --ulim_n) ulim_n="$2"; shift ;;
    --ns) ns="$2"; shift ;;
    --run_name) run_name="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -z "$robot_pkg_name" ]; then
  echo "Error: --robot_pkg_name is mandatory."
  usage
fi

# activate micromamba for this shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${MAMBA_ENV_NAME}

source /isaac-sim/setup_conda_env.sh
source $HOME/ibrido_ws/setup.bash

ulimit -n $ulim_n

reset && python $DIR1/launch_sim_env.py --headless --remote_stepping --robot_name $ns --robot_pkg_name $robot_pkg_name --num_envs $num_envs &
reset && python $DIR2/launch_control_cluster.py --ns $ns --size $num_envs & 
reset && python $DIR1/launch_train_env.py --ns $ns --run_name $run_name --drop_dir $HOME/training_data --dump_checkpoints --comment "" &

wait # wait for all to exit