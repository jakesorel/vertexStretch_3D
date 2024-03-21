#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=2-23:59:00   # walltime
#SBATCH -J "heart_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
source activate regression_modelling

python run_simulations.py ${SLURM_ARRAY_TASK_ID}

