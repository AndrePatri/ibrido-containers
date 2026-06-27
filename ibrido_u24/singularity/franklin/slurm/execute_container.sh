#!/bin/bash
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpua
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
#SBATCH --signal=B:SIGINT@300

set -euo pipefail

module load apptainer-1.4.1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SCHED_JOBID="${SLURM_JOB_ID:-${PBS_JOBID:-}}"

RUN_TOKEN="run_$(date +%Y%m%d_%H%M%S)_${SCHED_JOBID}"

echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
nvidia-smi

"${IBRIDO_CONTAINERS_PREFIX}/franklin/slurm/prescia_script.sh" "$1" "$RUN_TOKEN" &
"${IBRIDO_CONTAINERS_PREFIX}/execute.sh" --cfg "$1" --run_token "$RUN_TOKEN" --wdb_key "${WANDB_KEY}"

