#!/bin/bash
# SimCLR + 2-layer MLP Prediction Head — Full Pipeline
set -e
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."

ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

SAVE_DIR=./checkpoints/simclr_pred_2layers
CKPT=${SAVE_DIR}/checkpoint_0500.pth.tar

echo "=========================================="
echo "STEP 1: Pretrain SimCLR + 2-layer pred head (500 epochs, λ=0.5)"
echo "Started: $(date)"
echo "=========================================="
python main_simclr.py \
    --data ./imagenet100 \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 256 \
    --lr 0.3 \
    --temperature 0.5 \
    --pred-lambda 0.5 \
    --use-color \
    --use-rotation \
    --save-dir ${SAVE_DIR} \
    --workers 16

echo "=========================================="
echo "STEP 2: ImageNet-100 Object Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ${CKPT} \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 3: ImageNet-100 Rotation Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ${CKPT} \
    --eval-rotation \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 4: CUB-200 Object Eval (200 epochs)"
echo "Started: $(date)"
echo "=========================================="
python main_lincls.py \
    --data ../moco/cub200_prepared \
    --pretrained ${CKPT} \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160

echo "=========================================="
echo "STEP 5: Flowers-102 Few-shot Eval"
echo "Started: $(date)"
echo "=========================================="
python main_fewshot.py \
    --data ../flowers102_prepared \
    --pretrained ${CKPT}

echo "=========================================="
echo "ALL DONE: simclr_pred_2layers"
echo "Finished: $(date)"
echo "=========================================="
