#!/bin/bash
#SBATCH --job-name=muse_model_training
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/scratch/a/alim/afoninni/muse/benching_2.out
#SBATCH --error=/scratch/a/alim/afoninni/muse/benching_2.err


source /home/a/alim/afoninni/.virtualenvs/env3/bin/activate
cd /home/a/alim/afoninni/muse-apnea-detection/deep_learning || exit

export NUMBA_CACHE_DIR=/scratch/a/alim/afoninni/tmp  # fixes an error caused by a sub-dependency (numba)
export MPLCONFIGDIR=/scratch/a/alim/afoninni/tmp  # also a small fix (this one is not required)

export PYTHONPATH=..  # Set the working path directory to the project root so files like config can be accessed.

python apnea_binary_classifier.py