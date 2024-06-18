#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=1:ngpus=1
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q gpu_a100

export IBRIDO_CONTAINERS_PREFIX=""

$IBRIDO_CONTAINERS_PREFIX/build_singularity_isaac.sh --init --do_setup
