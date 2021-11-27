#!/bin/bash
BASE_DIR='./'
EXP_DIR="${BASE_DIR}/experiments/release/DTU/exp0"
cfg=./configs/dtu/release.conf

## train 
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train.py  --model_name=config0  --num_epochs=80 --DECAY_STEP_LIST 40 70 --cfg $cfg
