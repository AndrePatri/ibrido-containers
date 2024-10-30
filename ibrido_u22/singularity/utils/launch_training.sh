#!/bin/bash
WS_ROOT="$HOME/ibrido_ws"
LRHC_DIR="$WS_ROOT/src/LRHControl/lrhc_control/scripts"

usage() {
  echo "Usage: $0 --urdf_path URDF_PATH --srdf_path SRDF_PATH --jnt_imp_config_path JNT_IMP_CF_PATH \
  --cluster_client_fname CLUSTER_CL_FNAME [--num_envs NUM] [--set_ulim] [--ulim_n ULIM_N] \
    [--ns] [--run_name RUN_NAME] [--comment COMMENT] [--seed SEED] [--timeout_ms TIMEOUT] \
    [--codegen_override CG_OVERRIDE] \
    [--launch_rosbag LAUNCH_ROSBAG]\
    [--bag_sdt BAG_SDT] \
    [--ros_bridge_dt BRIDGE_DT] \
    [--dump_dt_min DUMP_DT] \
    [--env_idx_bag ENV_IDX_BAG] \
    [--custom_args_names CUSTOM_ARGS_NAMES] \
    [--custom_args_dtype CUSTOM_ARGS_DTYPE] \
    [--custom_args_vals CUSTOM_ARGS_VALS] \
    [--remote_stepping REMOTE_STEPPING] \
    "
  exit 1
}
num_envs=128
timeout_ms=60000
set_ulim=1
ulim_n=28672
seed=0
ns="ibrido"
run_name=""
comment=""
codegen_override=""
launch_rosbag_eval=false
bag_sdt=60.0
ros_bridge_dt=0.1
dump_dt_min=50.0
env_idx_bag=0
custom_args_names=""
custom_args_dtype=""
custom_args_vals=""
remote_stepping=1

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --urdf_path) urdf_path="$2"; shift ;;
    --srdf_path) srdf_path="$2"; shift ;;
    --jnt_imp_config_path) jnt_imp_config_path="$2"; shift ;;
    --cluster_client_fname) cluster_client_fname="$2"; shift ;;
    --num_envs) num_envs="$2"; shift ;;
    --timeout_ms) timeout_ms="$2"; shift ;;
    --set_ulim) set_ulim="$2"; shift ;;
    --ulim_n) ulim_n="$2"; shift ;;
    --ns) ns="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --run_name) run_name="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    --codegen_override) codegen_override="$2"; shift ;;
    --launch_rosbag) launch_rosbag="$2"; shift ;;
    --bag_sdt) bag_sdt="$2"; shift ;;
    --ros_bridge_dt) ros_bridge_dt="$2"; shift ;;
    --dump_dt_min) dump_dt_min="$2"; shift ;;
    --env_idx_bag) env_idx_bag="$2"; shift ;;
    --custom_args_names) custom_args_names="$2"; shift ;;
    --custom_args_dtype) custom_args_dtype="$2"; shift ;;
    --custom_args_vals) custom_args_vals="$2"; shift ;;
    --remote_stepping) remote_stepping="$2"; shift ;;
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

urdf_path_eval=$(eval echo $urdf_path)
srdf_path_eval=$(eval echo $srdf_path)
jnt_imp_config_path_eval=$(eval echo $jnt_imp_config_path)
cluster_client_fname_eval=$(eval echo $cluster_client_fname)
codegen_override_eval=$(eval echo $codegen_override)
launch_rosbag_eval=$(eval echo $launch_rosbag)
set_ulim_eval=$(eval echo $set_ulim)
remote_stepping$(eval echo $remote_stepping)

if (( set_ulim_eval )); then
  ulimit -n $ulim_n
fi

if (( remote_stepping )); then
  python $LRHC_DIR/launch_remote_env.py --headless --use_gpu --remote_stepping --robot_name $ns \
    --urdf_path $urdf_path_eval --srdf_path  $srdf_path_eval \
    --use_custom_jnt_imp --jnt_imp_config_path $jnt_imp_config_path_eval\
    --num_envs $num_envs --seed $seed --timeout_ms $timeout_ms \
    --custom_args_names $custom_args_names \
    --custom_args_dtype $custom_args_dtype \
    --custom_args_vals $custom_args_vals&
else
  python $LRHC_DIR/launch_remote_env.py --headless --use_gpu --robot_name $ns \
    --urdf_path $urdf_path_eval --srdf_path  $srdf_path_eval \
    --use_custom_jnt_imp --jnt_imp_config_path $jnt_imp_config_path_eval\
    --num_envs $num_envs --seed $seed --timeout_ms $timeout_ms \
    --custom_args_names $custom_args_names \
    --custom_args_dtype $custom_args_dtype \
    --custom_args_vals $custom_args_vals&
fi 

python $LRHC_DIR/launch_control_cluster.py --ns $ns --size $num_envs --timeout_ms $timeout_ms \
  --codegen_override_dir $codegen_override_eval \
  --cloop \
  --verbose \
  --urdf_path $urdf_path_eval --srdf_path $srdf_path_eval --cluster_client_fname $cluster_client_fname_eval \
  --custom_args_names $custom_args_names \
  --custom_args_dtype $custom_args_dtype \
  --custom_args_vals $custom_args_vals&

if (( remote_stepping )); then
  python $LRHC_DIR/launch_train_env.py --ns $ns --run_name $run_name --drop_dir $HOME/training_data --dump_checkpoints \
    --obs_norm --sac \
    --db --env_db --rmdb \
    --comment $comment --seed $seed --timeout_ms $timeout_ms &
fi 

if (( launch_rosbag_eval )); then
  if (( remote_stepping )); then
    python $LRHC_DIR/launch_periodic_bag_dump.py --ros2 --use_shared_drop_dir \
      --ns $ns --rhc_refs_in_h_frame \
      --bag_sdt $bag_sdt --ros_bridge_dt $ros_bridge_dt --dump_dt_min $dump_dt_min --env_idx $env_idx_bag --srdf_path $srdf_path_eval --with_agent_refs &
  else
    python $LRHC_DIR/launch_periodic_bag_dump.py --ros2 --use_shared_drop_dir \
      --ns $ns --rhc_refs_in_h_frame \
      --bag_sdt $bag_sdt --ros_bridge_dt $ros_bridge_dt --dump_dt_min $dump_dt_min --env_idx $env_idx_bag --srdf_path $srdf_path_eval &
  fi
fi

wait # wait for all to exit