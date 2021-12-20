#!/bin/bash

#SBATCH --exclude=kepler4,kepler3

module load python/3.9
virtualenv $HOME/py39

source $HOME/py39/bin/activate
pip install torch torchvision

python -c "import torch; print('is cuda available: ', torch.cuda.is_available())"
