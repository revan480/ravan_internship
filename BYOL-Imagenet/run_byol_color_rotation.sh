#!/bin/bash
# BYOL pre-training with color+rotation on ImageNet-100
# Run on britannia
# Usage: nohup bash run_byol_color_rotation.sh > byol_color_rotation.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
cd ~/projects/ravan_internship/BYOL-Imagenet

ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

python main_byol.py \
    --data ./imagenet100 \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 256 \
    --lr 0.3 \
    --tau-base 0.996 \
    --use-color \
    --use-rotation \
    --save-dir ./checkpoints/exp_byol_color_rotation \
    --workers 16
