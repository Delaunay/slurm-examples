#!/bin/bash

# Usage:
#   sbatch --nodes 1 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G scripts/multi-gpu.sh seedproject/train_normal.py
#


# Slurm configuration
# ===================

# we need all nodes to be ready at the same time
#SBATCH --wait-all-nodes=1
#SBATCH --partition=long
#SBATCH --exclude=kepler4,kepler3

# Total resources:
#   CPU: 16 * 1 = 16
#   RAM: 16 * 1 = 16 Go
#   GPU:  4 * 1 = 4

# Config
# ===================

# Setup our rendez-vous point
RDV_ADDR=localhost
WORLD_SIZE=$SLURM_JOB_NUM_NODES


# Setup
# ===================

module load python/3.7
module load python/3.7/cuda/11.1/cudnn/8.0/pytorch
source ~/envs/py37/bin/activate

export SEEDPROJECT_DATASET_PATH=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint

export GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

cmd="srun -l torchrun \
    --nproc_per_node=$GPU_COUNT\
    --nnodes=$WORLD_SIZE\
    --rdzv_id=$SLURM_JOB_ID\
    --rdzv_backend=c10d\
    --rdzv_endpoint=$RDV_ADDR\
    $@"

echo $cmd
$cmd
