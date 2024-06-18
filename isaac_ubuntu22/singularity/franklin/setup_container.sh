#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=1
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q gpu_a100
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/setup_cfg.sh 

$SCRIPT_DIR/../build_singularity_isaac.sh --init --do_setup