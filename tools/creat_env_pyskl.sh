#!/usr/bin/env bash

conda create -n pyskl python=3.9

conda activate pyskl

# pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/mmcv_full-1.5.0-cp37-cp37m-manylinux1_x86_64.whl

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install fvcore==0.1.5.post20221221

pip install yapf==0.32.0