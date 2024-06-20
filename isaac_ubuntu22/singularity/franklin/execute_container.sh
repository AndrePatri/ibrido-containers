#!/bin/bash
#PBS -l select=1:ncpus=32:mpiprocs=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q a100f

module load go-1.19.4/apptainer-1.1.8

export IBRIDO_CONTAINERS_PREFIX=""
export WANDB_KEY=""
export COMMENT=""

$IBRIDO_CONTAINERS_PREFIX/franklin/prescia_script.sh &

$IBRIDO_CONTAINERS_PREFIX/execute.sh --wandb_key $WANDB_KEY --comment $COMMENT
