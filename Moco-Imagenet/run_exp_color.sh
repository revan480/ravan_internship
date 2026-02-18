#!/bin/bash
# Experiment: Color augmentation (standard MoCo v2 baseline)
# --use-color (default), no rotation
set -e

DATA="./imagenet100"
SAVE_DIR="./checkpoints/exp_color"
LOG="exp_color.log"

echo "Starting exp_color (standard MoCo v2 baseline)" | tee ${LOG}
echo "Log: ${LOG}" | tee -a ${LOG}
date | tee -a ${LOG}

# Step 1: Pre-train MoCo (500 epochs)
echo "" | tee -a ${LOG}
echo ">>> Step 1: MoCo Pre-Training (500 epochs)" | tee -a ${LOG}
python main_moco.py \
    --data ${DATA} \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 256 \
    --moco-k 16384 \
    --save-dir ${SAVE_DIR} \
    --save-freq 50 \
    2>&1 | tee -a ${LOG}

CHECKPOINT="${SAVE_DIR}/checkpoint_0500.pth.tar"

# Step 2: Linear eval — object classification (100 classes)
echo "" | tee -a ${LOG}
echo ">>> Step 2: Linear Eval — Object Classification" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} \
    --arch resnet50 \
    --pretrained ${CHECKPOINT} \
    --epochs 200 \
    2>&1 | tee -a ${LOG}

# Step 3: Linear eval — rotation classification (4 classes)
echo "" | tee -a ${LOG}
echo ">>> Step 3: Linear Eval — Rotation Classification" | tee -a ${LOG}
python main_lincls.py \
    --data ${DATA} \
    --arch resnet50 \
    --pretrained ${CHECKPOINT} \
    --eval-rotation \
    --epochs 200 \
    2>&1 | tee -a ${LOG}

echo "" | tee -a ${LOG}
echo "exp_color COMPLETE" | tee -a ${LOG}
date | tee -a ${LOG}
