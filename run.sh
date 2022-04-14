#!/bin/sh
#SBATCH --time=100:00:00
#SBATCH --partition=normal_q
#SBATCH -n 32
#SBATCH --mem=100G
#SBATCH --account=aipmm

python src/main.py