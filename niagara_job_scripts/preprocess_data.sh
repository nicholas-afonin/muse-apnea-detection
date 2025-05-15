#!/bin/bash
#SBATCH --job-name=muse_preprocessing
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/scratch/a/alim/afoninni/muse/muse_preprocessing.out
#SBATCH --error=/scratch/a/alim/afoninni/muse/muse_preprocessing.err


source /home/a/alim/afoninni/.virtualenvs/myenv/bin/activate
cd /home/a/alim/afoninni/anne-apnea-detection || exit
python sequential_model_anne_preprocessing.py