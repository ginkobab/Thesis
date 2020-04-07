#!/bin/bash

sudo add-apt-repository ppa:nest-simulator/nest
sudo apt-get update
sudo apt-get install nest
sudo apt-get install python3-pip

pip3 install --upgrade pip
pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install matplotlib pandas seaborn numpy

