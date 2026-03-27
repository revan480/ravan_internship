#!/bin/bash
set -e
DATA=~/Desktop/ravan/flowers102_prepared

echo "=== MoCo+pred λ=0.5 — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_moco_pred_05/checkpoint_0500.pth.tar \
    --n-shots 5 10

echo "=== MoCo+pred λ=0.1 — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_moco_pred_01/checkpoint_0500.pth.tar \
    --n-shots 5 10
