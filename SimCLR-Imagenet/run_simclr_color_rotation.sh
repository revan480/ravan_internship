#!/bin/bash
# SimCLR pre-training with color+rotation on ImageNet-100
# Run on britannia with A6000

# Fix symlinks
ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

CUDA_VISIBLE_DEVICES=0 python main_simclr.py \
    --data ./imagenet100 \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 512 \
    --lr 0.3 \
    --temperature 0.5 \
    --use-color \
    --use-rotation \
    --save-dir ./checkpoints/exp_simclr_color_rotation \
    --workers 8
