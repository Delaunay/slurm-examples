#!/bin/bash


conda remove --name py39 --all

module load miniconda/3
conda create -n py39 python=3.9 -y
conda activate py39

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

python -c "import torch; print('is cuda available: ', torch.cuda.is_available())"
