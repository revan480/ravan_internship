#!/bin/bash
# Quick end-to-end test of LooC on imagenet100_tiny (5 classes, ~250 images).
# Runs in under 2 minutes on any GPU.
set -e

DATA="./looc/imagenet100_tiny"
SAVE_DIR="./checkpoints/test_tiny"

echo "=================================================="
echo "  LooC End-to-End Test (imagenet100_tiny)"
echo "=================================================="
date

# --------------------------------------------------
# Step 1: Syntax check
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  Step 1: Syntax check"
echo "=================================================="
python -c "from looc.builder import LooC; print('builder OK')"
python -c "from looc.loader import LooCTransform; print('loader OK')"

# --------------------------------------------------
# Step 2: LooC pre-training (2 epochs, tiny settings)
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  Step 2: LooC Pre-Training (2 epochs)"
echo "=================================================="
python main_looc.py \
    --data ${DATA} \
    --arch resnet50 \
    --epochs 2 \
    --batch-size 4 \
    --moco-k 64 \
    --moco-t 0.2 \
    --n-aug 2 \
    --workers 2 \
    --save-dir ${SAVE_DIR} \
    --save-freq 1 \
    --print-freq 1

# --------------------------------------------------
# Step 3: Verify checkpoint contents
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  Step 3: Verify checkpoint contents"
echo "=================================================="
python -c "
import torch
ck = torch.load('${SAVE_DIR}/checkpoint_0002.pth.tar', map_location='cpu')
keys = list(ck['state_dict'].keys())
print(f'Total keys: {len(keys)}')
for k in keys:
    if any(x in k for x in ['backbone_q.conv1', 'backbone_q.layer1.0.conv1', 'heads_q.0.0', 'heads_q.1.0', 'heads_q.2.0', 'queue_0', 'queue_1', 'queue_2', 'queue_ptr']):
        print(k)
"

# --------------------------------------------------
# Step 4: Linear eval — object classification (2 epochs)
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  Step 4: Linear Eval — Object Classification (2 epochs)"
echo "=================================================="
python main_lincls.py \
    --data ${DATA} \
    --pretrained ${SAVE_DIR}/checkpoint_0002.pth.tar \
    --looc-backbone \
    --epochs 2 \
    --batch-size 4 \
    --workers 2 \
    --print-freq 1

# --------------------------------------------------
# Step 5: Linear eval — rotation classification (2 epochs)
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  Step 5: Linear Eval — Rotation Classification (2 epochs)"
echo "=================================================="
python main_lincls.py \
    --data ${DATA} \
    --pretrained ${SAVE_DIR}/checkpoint_0002.pth.tar \
    --looc-backbone \
    --eval-rotation \
    --epochs 2 \
    --batch-size 4 \
    --workers 2 \
    --print-freq 1

# --------------------------------------------------
# Step 6: Done
# --------------------------------------------------
echo ""
echo "=================================================="
echo "  ALL TESTS PASSED"
echo "=================================================="
date
