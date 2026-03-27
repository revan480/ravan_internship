#!/bin/bash
set -e
DATA=~/Desktop/ravan/flowers102_prepared

echo "=== MoCo color — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_color/checkpoint_0500.pth.tar \
    --n-shots 5 10

echo "=== MoCo rotation — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_rotation/checkpoint_0500.pth.tar \
    --n-shots 5 10

echo "=== MoCo color+rotation — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_color_rotation/checkpoint_0500.pth.tar \
    --n-shots 5 10
