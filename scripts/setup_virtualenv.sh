#!/bin/bash

#SBATCH --exclude=kepler4,kepler3

module load python/3.8
virtualenv $HOME/py38

source $HOME/py38/bin/activate
pip install torch torchvision

python -c "import torch; print('is cuda available: ', torch.cuda.is_available())"
