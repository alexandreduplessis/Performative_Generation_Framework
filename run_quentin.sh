#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

#SBATCH --exclude=cn-e002,cn-e003,cn-b001,cn-b002,cn-b003,cn-b004,cn-b005,kepler[5],cn-g[001-026]

#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt


module load anaconda/3
conda activate torchopt

python expes/main.py
