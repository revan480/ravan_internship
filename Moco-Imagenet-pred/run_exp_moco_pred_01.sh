#!/bin/bash
set -e
DATA="./imagenet100"
SAVE_DIR="./checkpoints/exp_moco_pred_01"
LOG="exp_moco_pred_01.log"
CHECKPOINT="${SAVE_DIR}/checkpoint_0500.pth.tar"

mkdir -p ${SAVE_DIR}

echo "Starting MoCo + Aug Prediction (lambda=0.1)" | tee ${LOG}
date | tee -a ${LOG}

echo ">>> Step 1: Pre-Training (500 epochs)" | tee -a ${LOG}
python main_moco.py \
    --data ${DATA} --arch resnet50 --epochs 500 \
    --batch-size 256 --lr 0.03 --schedule 300 400 \
    --moco-k 16384 --moco-t 0.2 --workers 12 \
    --use-color --use-rotation \
    --lambda-pred 0.1 \
    --save-dir ${SAVE_DIR} --save-freq 50 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 2: Object Classification Eval" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --batch-size 256 --workers 12 --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 3: Rotation Classification Eval" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --eval-rotation \
    --batch-size 256 --workers 12 --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo "COMPLETE" | tee -a ${LOG}
date | tee -a ${LOG}
