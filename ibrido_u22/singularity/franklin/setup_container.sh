#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=1:ngpus=1
#PBS -l walltime=00:60:00
#PBS -j oe
#PBS -N ibrido_build
#PBS -q a100f

module load go-1.19.4/apptainer-1.1.8

$IBRIDO_CONTAINERS_PREFIX/build_singularity_isaac.sh --init --do_setup
