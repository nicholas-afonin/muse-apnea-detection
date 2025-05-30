#!/bin/bash
#SBATCH --job-name=muse_preprocessing
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/scratch/a/alim/afoninni/muse/eeg-features-10s.out
#SBATCH --error=/scratch/a/alim/afoninni/muse/eeg-features-10s.err


source /home/a/alim/afoninni/.virtualenvs/env2/bin/activate
cd /home/a/alim/afoninni/muse-apnea-detection/preprocessing || exit

export NUMBA_CACHE_DIR=/scratch/a/alim/afoninni/tmp  # fixes an error caused by a sub-dependency (numba)
export MPLCONFIGDIR=/scratch/a/alim/afoninni/tmp  # also a small fix (this one is not required)
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python EEG_features_April_2025.py