#!/bin/bash
#SBATCH --partition=savio2
#SBATCH --time=30
#SBATCH --tasks-per-node=28
#SBATCH --account=fc_ceder
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --job-name=analyze_elemnet
#SBATCH --qos=savio_debug

python analyze.py
