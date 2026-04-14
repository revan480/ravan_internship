#!/bin/bash
# BYOL Full Pipeline — Pre-training + All Evaluations
# Run on britannia
# Usage: nohup bash run_byol_full_pipeline.sh > byol_full_pipeline.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
cd ~/projects/ravan_internship/BYOL-Imagenet

ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

echo "=========================================="
echo "STEP 1: BYOL Pre-training (500 epochs)"
echo "Started: $(date)"
echo "=========================================="
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

echo "=========================================="
echo "STEP 2: ImageNet-100 Object Eval"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ./checkpoints/exp_byol_color_rotation/checkpoint_0500.pth.tar \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 3: ImageNet-100 Rotation Eval"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ./checkpoints/exp_byol_color_rotation/checkpoint_0500.pth.tar \
    --eval-rotation \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 4: CUB-200 Object Eval"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ../moco/cub200_prepared \
    --pretrained ./checkpoints/exp_byol_color_rotation/checkpoint_0500.pth.tar \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 5: Flowers-102 Few-shot Eval"
echo "Started: $(date)"
echo "=========================================="
python main_fewshot.py \
    --data ../flowers102_prepared \
    --pretrained ./checkpoints/exp_byol_color_rotation/checkpoint_0500.pth.tar

echo "=========================================="
echo "ALL DONE!"
echo "Finished: $(date)"
echo "=========================================="
