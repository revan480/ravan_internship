#!/bin/bash
set -e
DATA="./imagenet100"

echo "=== LooC++ eval on existing angle-pred λ=0.5 checkpoint ==="

echo ">>> LooC++ Object eval"
python main_lincls.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_cr_pred_05/checkpoint_0500.pth.tar \
    --looc-plus --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160

echo ">>> LooC++ Rotation eval"
python main_lincls.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_cr_pred_05/checkpoint_0500.pth.tar \
    --looc-plus --eval-rotation --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160

echo "=== LooC++ eval on existing LooC baseline (no pred) checkpoint ==="

echo ">>> LooC++ Object eval"
python main_lincls.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color_rotation/checkpoint_0500.pth.tar \
    --looc-plus --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160

echo ">>> LooC++ Rotation eval"
python main_lincls.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color_rotation/checkpoint_0500.pth.tar \
    --looc-plus --eval-rotation --batch-size 256 --workers 12 \
    --epochs 200 --lr 30.0 --schedule 120 160

echo "COMPLETE"
