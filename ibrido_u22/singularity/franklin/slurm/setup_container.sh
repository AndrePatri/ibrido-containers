#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=ibrido_run
#SBATCH --partition=gpua
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

module load apptainer-1.4.1

$IBRIDO_CONTAINERS_PREFIX/build_singularity_isaac.sh --init --do_setup --ngc $NGC_KEY
