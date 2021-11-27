#!/bin/bash
BASE_DIR='./'
cfg=./configs/dtu/release.conf

EXP_DIR=./experiments/release/DTU/exp0
CUDA_VISIBLE_DEVICES=0  python  train.py  --model_name=config0_test --mode=full_test --cfg $cfg --multiprocessing_distributed 0 --save_prediction 1 --log_dir=$EXP_DIR --load_weights_folder=./pretrained_model/dtu/MVS2D

