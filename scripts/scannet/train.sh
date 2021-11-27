#!/bin/bash
BASE_DIR='./'
EXP_DIR="${BASE_DIR}/experiments/release/ScanNet/exp0"
cfg=./configs/scannet/release.conf

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train.py  --num_epochs=30 --DECAY_STEP_LIST 25 28 --cfg $cfg  --fullsize_eval=1 --use_test=0
