#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpua
# Join stdout and stderr into a single file (like PBS -j oe)
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
#SBATCH --signal=B:SIGINT@300

module load apptainer-1.4.1

# One thread per process, NOT one per allocated CPU. See franklin/slurm/execute_container.sh for
# the full rationale: the MPC cluster runs one process per env (800 for a standard run), each of
# which links libgomp through libipopt. Left unset, libgomp defaults to the cgroup's CPU count
# (here 32), giving 800 x 32 = 25600 threads on a 32-CPU allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export SCHED_JOBID="${SLURM_JOB_ID:-${PBS_JOBID:-}}"

# Stop sentinel, shared with prescia_script.sh and execute_ablation.sh. An ablation runs its configs
# sequentially, so when prescia fires near the walltime it must both (a) SIGINT the run in flight so
# it saves cleanly and (b) stop the loop from starting another one that would be hard-killed.
export IBRIDO_STOP_FILE="/tmp/ibrido_ablation_stop_${SCHED_JOBID:-$$}"
rm -f "$IBRIDO_STOP_FILE"

# NOTE: this used to point at franklin/pbs/prescia_script.sh, which DOES NOT EXIST -- so the ablation
# silently ran with no deadline handling at all and was hard-killed at the walltime, losing the model
# and the debug dump of whatever run was in flight. Use the slurm one, with no run token so it falls
# back to the generic 'execute.sh --cfg' pattern (the config changes on every iteration).
"$IBRIDO_CONTAINERS_PREFIX/franklin/slurm/prescia_script.sh" "" "" &
prescia_pid=$!

"$IBRIDO_CONTAINERS_PREFIX/execute_ablation.sh" --cfg_dir "$1"

kill "$prescia_pid" 2>/dev/null || true
rm -f "$IBRIDO_STOP_FILE"
