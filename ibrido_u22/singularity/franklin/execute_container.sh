#!/bin/bash
#PBS -l select=1:ncpus=32:mpiprocs=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ibrido_run
#PBS -q gpu_a100

module load go-1.19.4/apptainer-1.1.8

export PBS_JOBID="${PBS_JOBID}"

$IBRIDO_CONTAINERS_PREFIX/franklin/prescia_script.sh &
$IBRIDO_CONTAINERS_PREFIX/execute.sh --cfg {}
