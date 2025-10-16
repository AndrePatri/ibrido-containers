#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpu_a100
# Join stdout and stderr into a single file (like PBS -j oe)
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

module load go-1.19.4/apptainer-1.1.8

export SCHED_JOBID="${SLURM_JOB_ID:-$PBS_JOBID}"

$IBRIDO_CONTAINERS_PREFIX/franklin/pbs/prescia_script.sh &
$IBRIDO_CONTAINERS_PREFIX/execute_ablation.sh --cfg_dir $cfg
