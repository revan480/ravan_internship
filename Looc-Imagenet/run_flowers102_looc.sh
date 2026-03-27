#!/bin/bash
set -e
DATA=~/Desktop/ravan/flowers102_prepared

echo "=== LooC color+rotation — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color_rotation/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10

echo "=== LooC rotation — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_rotation/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10

echo "=== LooC color — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10
