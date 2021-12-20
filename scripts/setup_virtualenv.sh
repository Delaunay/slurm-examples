#!/bin/bash

module load python/3.9
virtualenv $HOME/py39

source $HOME/py39/bin/activate
pip install torch torchvision

python -c "import torch; print(torch.cuda.is_available())"
