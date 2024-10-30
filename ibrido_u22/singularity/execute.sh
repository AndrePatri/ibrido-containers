#!/bin/bash
# set -e # exiting if any cmd fails

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"
source "${IBRIDO_CONTAINERS_PREFIX}/files/training_cfg.sh"

# Function to print usage
usage() {
    echo "Usage: $0 [--use_sudo|-s] [--set_ulim|-ulim]"
    exit 1
}
use_sudo=false # whether to use superuser privileges
set_ulim=false

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--use_sudo) use_sudo=true ;;
        -ulim|--set_ulim) set_ulim=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# convert bind dirs to comma-separated list
IFS=',' # Set the internal field separator to a comma
binddirs="${IBRIDO_B_ALL[*]}"
unset IFS # Reset the internal field separator

training_script="launch_training.sh"

singularity_cmd="singularity exec \
    --env \"WANDB_KEY=$WANDB_KEY\"\
    --env \"ROS_LOCALHOST_ONLY=1\"\
    --bind $binddirs\
    --no-mount home,cwd \
    --nv $IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif $training_script \
    --urdf_path $URDF_PATH \
    --srdf_path $SRDF_PATH \
    --jnt_imp_config_path $JNT_IMP_CF_PATH \
    --cluster_client_fname $CLUSTER_CL_FNAME \
    --num_envs $N_ENVS \
    --set_ulim $SET_ULIM\
    --ulim_n $ULIM_N \
    --ns $SHM_NS \
    --run_name $RNAME \
    --comment $COMMENT \
    --seed $SEED \
    --timeout_ms $TIMEOUT_MS \
    --codegen_override $CODEGEN_OVERRIDE_BDIR \
    --launch_rosbag $LAUNCH_ROSBAG \
    --bag_sdt $BAG_SDT \
    --ros_bridge_dt $BRIDGE_DT \
    --dump_dt_min $DUMP_DT \
    --env_idx_bag $ENV_IDX_BAG"

if $use_sudo; then
    sudo bash -c "$singularity_cmd"
else
    bash -c "$singularity_cmd"
fi

