#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpu_a100
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

module load go-1.19.4/apptainer-1.1.8

$IBRIDO_CONTAINERS_PREFIX/build_singularity_isaac.sh --init --do_setup --ngc $NGC_KEY
