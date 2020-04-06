#!/bin/bash

apt-get install python3-venv
add-apt-repository ppa:nest-simulator/nest
apt-get update
apt-get install nest

python3 -m venv ~/.virtualenvs/thesis
source ~/.virtualenvs/thesis/bin/activate

pip install --upgrade pip
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib pandas seaborn numpy

