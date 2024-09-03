#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"

usage() {
  echo "Usage: $0 --urdf_path URDF_PATH --srdf_path SRDF_PATH --jnt_imp_config_path JNT_IMP_CF_PATH \
  --cluster_client_fname CLUSTER_CL_FNAME [--num_envs NUM] [--set_ulim|-ulim] [--ulim_n ULIM_N] \
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
    --urdf_path) urdf_path="$2"; shift ;;
    --srdf_path) srdf_path="$2"; shift ;;
    --jnt_imp_config_path) jnt_imp_config_path="$2"; shift ;;
    --cluster_client_fname) cluster_client_fname="$2"; shift ;;
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

if [ -z "$urdf_path" ]; then
  echo "Error: --urdf_path is mandatory."
  usage
fi

if [ -z "$srdf_path" ]; then
  echo "Error: --srdf_path is mandatory."
  usage
fi

if [ -z "$jnt_imp_config_path" ]; then
  echo "Error: --jnt_imp_config_path is mandatory."
  usage
fi

if [ -z "$cluster_client_fname" ]; then
  echo "Error: --cluster_client_fname is mandatory."
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

urdf_path_eval=$(eval echo $urdf_path)
srdf_path_eval=$(eval echo $srdf_path)
jnt_imp_config_path_eval=$(eval echo $jnt_imp_config_path)
cluster_client_fname_eval=$(eval echo $cluster_client_fname)
codegen_override_eval=$(eval echo $codegen_override)

python $LRHC_DIR/launch_remote_env.py --headless --remote_stepping --robot_name $ns \
 --urdf_path $urdf_path_eval --srdf_path  $srdf_path_eval --jnt_imp_config_path $jnt_imp_config_path_eval\
 --num_envs $num_envs --timeout_ms $timeout_ms&
python $LRHC_DIR/launch_control_cluster.py --ns $ns --size $num_envs --timeout_ms $timeout_ms \
  --codegen_override_dir $codegen_override_eval \
  --urdf_path $urdf_path_eval --srdf_path $srdf_path_eval --cluster_client_fname $cluster_client_fname_eval & 
python $LRHC_DIR/launch_train_env.py --ns $ns --run_name $run_name --drop_dir $HOME/training_data --dump_checkpoints \
  --comment $comment --seed $seed --timeout_ms $timeout_ms&

wait # wait for all to exit