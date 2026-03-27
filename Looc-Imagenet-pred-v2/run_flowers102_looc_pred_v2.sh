#!/bin/bash
set -e
DATA=~/Desktop/ravan/flowers102_prepared

echo "=== LooC+combo-pred λ=0.5 — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_combo_pred_05/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10

echo "=== LooC+combo-pred λ=0.5 (LooC++) — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_combo_pred_05/checkpoint_0500.pth.tar \
    --looc-plus --n-shots 5 10

echo "=== LooC+pred λ=0.5 — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_cr_pred_05/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10

echo "=== LooC+pred λ=0.5 (LooC++) — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_cr_pred_05/checkpoint_0500.pth.tar \
    --looc-plus --n-shots 5 10

echo "=== LooC color+rotation baseline — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color_rotation/checkpoint_0500.pth.tar \
    --looc-backbone --n-shots 5 10

echo "=== LooC color+rotation baseline (LooC++) — Flowers-102 ==="
python main_fewshot.py \
    --data ${DATA} \
    --pretrained ./checkpoints/exp_looc_color_rotation/checkpoint_0500.pth.tar \
    --looc-plus --n-shots 5 10
