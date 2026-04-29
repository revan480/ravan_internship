#!/bin/bash
# E-SSL Figure 1: vanilla SimCLR baseline
set -e
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."

ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

SAVE_DIR=./checkpoints/baseline
CKPT=${SAVE_DIR}/checkpoint_0150.pth.tar

echo "=========================================="
echo "STEP 1: Pretrain baseline / vanilla SimCLR (150 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_simclr.py \
    --baseline \
    --data ./imagenet100 \
    --arch resnet18 \
    --epochs 150 \
    --batch-size 256 \
    --lr 0.3 \
    --temperature 0.5 \
    --save-dir ${SAVE_DIR} \
    --workers 16

echo "=========================================="
echo "STEP 2: ImageNet-100 Linear Eval"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ${CKPT} \
    --arch resnet18 \
    --batch-size 256 --epochs 100 --lr 30.0 --schedule 60 80

echo "=========================================="
echo "STEP 3: CUB-200 Linear Eval"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ../moco/cub200_prepared \
    --pretrained ${CKPT} \
    --arch resnet18 \
    --batch-size 256 --epochs 100 --lr 30.0 --schedule 60 80

echo "=========================================="
echo "STEP 4: Flowers-102 Few-shot Eval"
echo "Started: $(date)"
echo "=========================================="
python main_fewshot.py \
    --data ../flowers102_prepared \
    --pretrained ${CKPT} \
    --arch resnet18

echo "=========================================="
echo "ALL DONE: baseline / vanilla SimCLR"
echo "Finished: $(date)"
echo "=========================================="
