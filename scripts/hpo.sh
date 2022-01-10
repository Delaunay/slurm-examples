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

export EXPERIMENT_NAME="seedexperiment"


# Constant
export SCRATCH=~/scratch
export SEEDPROJECT_DATASET_DEST=$SLURM_TMPDIR/dataset
export SEEDPROJECT_CHECKPOINT_PATH=~/scratch/checkpoint
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
export SPACE_CONFIG=$SCRATCH/space-config_${SEQ}.json

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
        name: ${EXPERIMENT_NAME}_${SEQ}
        algorithms:
            hyperband:
                seed: null
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
EOM

cat > $SPACE_CONFIG <<- EOM
    {
        "epochs": "orion~fidelity(1, 100, base=2)",
        "lr": "orion~loguniform(1e-5, 1.0)",
        "weight_decay": "orion~loguniform(1e-10, 1e-3)",
        "momentum": "orion~loguniform(0.9, 1.0)"
    }
EOM


# Run
# ===================

cmd="orion hunt --config $ORION_CONFIG python $@ --config $SPACE_CONFIG"

echo $cmd
$cmd
