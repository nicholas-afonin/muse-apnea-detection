#!/bin/bash
#SBATCH --job-name=muse_preprocessing
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/scratch/a/alim/afoninni/muse/muse_preprocessing.out
#SBATCH --error=/scratch/a/alim/afoninni/muse/muse_preprocessing.err


source /home/a/alim/afoninni/.virtualenvs/env2/bin/activate
cd /home/a/alim/afoninni/muse-apnea-detection/preprocessing || exit
python combine_features.py