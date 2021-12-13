#!/bin/bash

# Usage:
#   sbatch --gres=gpu:1 --cpus-per-gpu=4 --mem=16G scripts/single-gpu.sh seedproject/train_normal.py
#

# Setup our rendez-vous point
RDV_ADDR=localhost
WORLD_SIZE=$SLURM_JOB_NUM_NODES
# -----

module load miniconda/3
conda activate py39

export SEEDPROJECT_DATASET_PATH=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint

cmd="$@"

echo $cmd

python $cmd
