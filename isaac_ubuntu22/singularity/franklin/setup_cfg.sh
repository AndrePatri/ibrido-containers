#!/bin/bash

#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=1
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q gpu_a100

WANDB_KEY="#######"
