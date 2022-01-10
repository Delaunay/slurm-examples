#!/bin/bash
set -evx

# Usage:
#   sbatch --nodes 3 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G multi-node.sh training_script.py
#

# Slurm configuration
# ===================

# we need all nodes to be ready at the same time
#SBATCH --wait-all-nodes=1
#SBATCH --partition=long
#SBATCH --exclude=kepler4,kepler3

# Total resources:
#   CPU: 16 * 3 = 48
#   RAM: 16 * 3 = 48 Go
#   GPU:  4 * 3 = 12


# Python
# ===================

module load miniconda/3
conda activate py39

# Environment
# ===================

# Setup our rendez-vous point
RDV_ADDR=$(hostname)
#                 $SLURM_NNODES
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

#                $SLURM_GPUS_PER_NODE       => not set
#                $SLURM_GPUS_ON_NODE        => not set
#                $SLURM_GPUS                => not set
#                $SLURM_JOB_GPUS=1,2,3,5
#                $GPU_DEVICE_ORDINAL=0,1,2,3
#                $CUDA_VISIBLE_DEVICES=0,1,2,3
# export GPU_COUNT=$(python -c "import os; print(len(os.environ['GPU_DEVICE_ORDINAL'].split(',')))") 
# export GPU_COUNT=$(($SLURM_CPUS_ON_NODE / $SLURM_CPUS_PER_GPU))

export GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")  

#                      $SLURM_CPUS_ON_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

#             $SLURM_MEM_PER_NODE
export MEMORY=$SLURM_MEM_PER_GPU

export SEEDPROJECT_DATASET_DEST=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint


# Run
# ===================

srun -l torchrun \
    --nproc_per_node=$GPU_COUNT\
    --nnodes=$WORLD_SIZE\
    --rdzv_id=$SLURM_JOB_ID\
    --rdzv_backend=c10d\
    --rdzv_endpoint=$RDV_ADDR\
    "$@"
