# salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:30:00 --partition=gpua --gres=gpu:1

# sbatch execute_container.sh runs/training/centauro/centauro_ub_pert_recovery_isaac5x.yaml

# squeue -u $USER

# scontrol show job $JOB_ID

# squeue -w afnode01

# srun --jobid=<jobid> --ntasks=1 --pty bash --> connect an interactive session

# pkill -SIGINT -f launch_train_env.py

# squeue --start --job $JOB_ID

# squeue -j $JOB_ID -o "%.18i %.2t %.10M %.10l %.10L"
