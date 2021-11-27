#!/bin/bash
BASE_DIR='./'
EXP_DIR="${BASE_DIR}/experiments/release/DeMoN/exp0"
cfg=./configs/demon/release.conf

CUDA_VISIBLE_DEVICES=0  python  train.py  --model_name=config0_test --mode=test --cfg $cfg  --use_test=1 --fullsize_eval=1 --load_weights_folder=./pretrained_model/demon/MVS2D
