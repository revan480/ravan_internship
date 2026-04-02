#!/bin/bash
# SimCLR Full Pipeline — Pre-training + All Evaluations
# Designed to run unattended on britannia GPU 1
# Usage: nohup bash run_full_pipeline.sh > simclr_full_pipeline.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
cd ~/projects/ravan_internship/SimCLR-Imagenet

# Ensure symlinks
ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

echo "=========================================="
echo "STEP 1: SimCLR Pre-training (500 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_simclr.py \
    --data ./imagenet100 \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 256 \
    --lr 0.3 \
    --temperature 0.5 \
    --use-color \
    --use-rotation \
    --save-dir ./checkpoints/exp_simclr_color_rotation \
    --workers 4

echo "=========================================="
echo "STEP 2: ImageNet-100 Object Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ./checkpoints/exp_simclr_color_rotation/checkpoint_0500.pth.tar \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 3: ImageNet-100 Rotation Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ./checkpoints/exp_simclr_color_rotation/checkpoint_0500.pth.tar \
    --eval-rotation \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 4: CUB-200 Object Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ../moco/cub200_prepared \
    --pretrained ./checkpoints/exp_simclr_color_rotation/checkpoint_0500.pth.tar \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 5: Flowers-102 Few-shot Eval"
echo "Started: $(date)"
echo "=========================================="
python main_fewshot.py \
    --data ../flowers102_prepared \
    --pretrained ./checkpoints/exp_simclr_color_rotation/checkpoint_0500.pth.tar

echo "=========================================="
echo "ALL DONE!"
echo "Finished: $(date)"
echo "=========================================="
