# salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:30:00 --partition=gpua --gres=gpu:1

# sbatch execute_container.sh centauro/training_cfg_centauro_ub_cloop_pert_recovery.sh

# squeue -u $USER

# scontrol show job $JOB_ID

# squeue -w afnode01
