#!/bin/bash
BASE_DIR='./'
EXP_DIR="${BASE_DIR}/experiments/release/DeMoN/exp0"
cfg=./configs/demon/release.conf

## train
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train.py  --num_epochs=30 --DECAY_STEP_LIST 25 28 --cfg $cfg --use_test=1
