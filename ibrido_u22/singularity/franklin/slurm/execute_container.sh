#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpua
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
#SBATCH --signal=B:SIGINT@300    # match prescia 5-minute early stop

module load apptainer-1.4.1

export SCHED_JOBID="${SLURM_JOB_ID:-$PBS_JOBID}"

# $IBRIDO_CONTAINERS_PREFIX/franklin/slurm/prescia_script.sh &
$IBRIDO_CONTAINERS_PREFIX/execute.sh --cfg "$1"
