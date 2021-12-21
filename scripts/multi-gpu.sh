#!/bin/bash
set -evx

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

# Python
# ===================
module load miniconda/3
conda activate py39

# Environment
# ===================

# Setup our rendez-vous point
RDV_ADDR=localhost

export WORLD_SIZE=$SLURM_JOB_NUM_NODES

#                $SLURM_GPUS_ON_NODE
export GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")  

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

export SEEDPROJECT_DATASET_PATH=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint


# Run
# ===================

cmd="srun -l torchrun \
    --nproc_per_node=$GPU_COUNT\
    --nnodes=$WORLD_SIZE\
    --rdzv_id=$SLURM_JOB_ID\
    --rdzv_backend=c10d\
    --rdzv_endpoint=$RDV_ADDR\
    $@"

echo $cmd
$cmd
