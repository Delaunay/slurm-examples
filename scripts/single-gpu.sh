#!/bin/bash


# Usage:
#   sbatch --gres=gpu:1 --cpus-per-gpu=4 --mem=16G scripts/single-gpu.sh seedproject/train_normal.py
#

# Slurm configuration
# ===================
#SBATCH --exclude=kepler4,kepler3


# Setup
# ===================

module load python/3.8
source $HOME/py38/bin/activate

export SEEDPROJECT_DATASET_PATH=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint

cmd="$@"

echo $cmd

python $cmd
