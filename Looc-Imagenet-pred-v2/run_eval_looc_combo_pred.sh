#!/bin/bash
set -e
DATA="./imagenet100"
SAVE_DIR="./checkpoints/exp_looc_combo_pred_05"
LOG="exp_looc_combo_pred_05.log"
CHECKPOINT="${SAVE_DIR}/checkpoint_0500.pth.tar"

mkdir -p ${SAVE_DIR}

echo "Starting LooC + Combo Prediction (lambda=0.5)" | tee ${LOG}
date | tee -a ${LOG}

echo ">>> Step 1: Pre-Training (500 epochs)" | tee -a ${LOG}
python main_looc.py \
    --data ${DATA} --arch resnet50 --epochs 500 \
    --batch-size 256 --lr 0.03 --schedule 300 400 \
    --moco-k 16384 --moco-t 0.2 --workers 12 \
    --aug-types rotation color \
    --lambda-pred 0.5 --pred-mode combo \
    --save-dir ${SAVE_DIR} --save-freq 50 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 2: LooC eval — Object" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --looc-backbone --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 3: LooC eval — Rotation" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --looc-backbone --eval-rotation --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 4: LooC++ eval — Object" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --looc-plus --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo ">>> Step 5: LooC++ eval — Rotation" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} --pretrained ${CHECKPOINT} \
    --looc-plus --eval-rotation --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160 \
    2>&1 | tee -a ${LOG}

echo "COMPLETE" | tee -a ${LOG}
date | tee -a ${LOG}
