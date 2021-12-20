#!/bin/bash

#SBATCH --exclude=kepler4,kepler3

module load miniconda/3
conda remove --name py39 --all -y

conda create -n py39 python=3.9 -y
conda activate py39

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y

python -c "import torch; print('is cuda available: ', torch.cuda.is_available())"
