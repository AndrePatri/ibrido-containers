#!/bin/bash
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q gpu_a100

module load go-1.19.4/apptainer-1.1.8

export IBRIDO_CONTAINERS_PREFIX=""
export WANDB_KEY=""
export COMMENT=""

$IBRIDO_CONTAINERS_PREFIX/execute.sh --wandb_key $WANDB_KEY --comment $COMMENT
