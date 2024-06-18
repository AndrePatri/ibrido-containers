#!/bin/bash
#PBS -l select=1:ncpus=96:mpiprocs=20:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q gpu_a100

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/run_cfg.sh

$SCRIPT_DIR/../execute.sh --wandb_key $WANDB_KEY --comment $COMMENT