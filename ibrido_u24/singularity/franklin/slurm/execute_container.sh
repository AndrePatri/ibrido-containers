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

# One thread per process, NOT one per allocated CPU.
#
# The MPC cluster spawns ONE PROCESS PER ENV (control_cluster_client.use_core_pool defaults to
# False, and IBRIDO never enables it), so an N_ENVS=800 run is 800 concurrent controller processes.
# libipopt.so -- which every one of them runs -- links libgomp + libblas + liblapack, so each
# process honours OMP_NUM_THREADS and will spawn that many threads inside its KKT solve.
#
# With the previous OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}=48 that is
#     800 controllers x 48 threads = 38400 threads on a 48-CPU allocation  (800:1)
# instead of the intended 800 processes / 48 CPUs (16.7:1) -- pure context-switch thrash.
# Parallelism here comes from having many processes, not from threads inside each one.
#
# Safe for the training process too: it is GPU-bound (nets, replay and Genesis all on device), so
# its CPU thread count is irrelevant.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export SCHED_JOBID="${SLURM_JOB_ID:-${PBS_JOBID:-}}"

RUN_TOKEN="run_$(date +%Y%m%d_%H%M%S)_${SCHED_JOBID}"

echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}  (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
nvidia-smi

"${IBRIDO_CONTAINERS_PREFIX}/franklin/slurm/prescia_script.sh" "$1" "$RUN_TOKEN" &
"${IBRIDO_CONTAINERS_PREFIX}/execute.sh" --cfg "$1" --run_token "$RUN_TOKEN" --wdb_key "${WANDB_KEY}"

