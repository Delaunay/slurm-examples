#!/bin/bash
set -v

# Slurm configuration
# ===================

#SBATCH --ntasks=1
#SBATCH --exclude=kepler4,kepler3


# Python
# ===================

module load miniconda/3
conda activate py39

# Environment
# ===================

export SCRATCH=/network/scratch/
export EXPERIMENT_NAME='MySuperExperiment'
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
export SEEDPROJECT_DATASET_PATH=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint

# Configure Orion
# ===================
# 
#    - user hyperband
#    - launch 4 workers for each tasks (one for each CPU)
#    - worker dies if idle for more than a minute
#    - Each worker are sharing a single GPU to maximize usage
#
cat > $ORION_CONFIG <<- EOM
    experiment:
        algorithms:
            hyperband:
                seed: None
        max_broken: 10

    worker:
        n_workers: $SBATCH_CPUS_PER_GPU
        pool_size: 0
        executor: joblib
        heartbeat: 120
        max_broken: 10
        idle_timeout: 60

    database:
        host: $SCRATCH/${EXPERIMENT_NAME}_orion.pkl
        type: pickleddb

    lr: 'orion~loguniform(1e-5, 1.0)'
    weight_decay: 'orion~loguniform(1e-10, 1e-3)'
    momentum: 'orion~loguniform(0.9, 1)'
EOM


# Run
# ===================

orion hunt --config $ORION_CONFIG python "$@"
