#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=240
#SBATCH --ntasks=36
#SBATCH --account=sngmd
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --job-name=analyze_elemnet

python analyze.py
