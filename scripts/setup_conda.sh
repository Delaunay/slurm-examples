#!/bin/bash

module load miniconda/3
conda create -n py39 python=3.9
conda activate py39

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

python -c "import torch; print(torch.cuda.is_available())"
