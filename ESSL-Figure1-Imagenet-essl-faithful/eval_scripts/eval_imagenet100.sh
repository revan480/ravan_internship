#!/bin/bash
# Linear probe on ImageNet-100
# Usage: bash eval_scripts/eval_imagenet100.sh <checkpoint_path>
set -e
CKPT=${1:?Usage: eval_imagenet100.sh <checkpoint_path>}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."

ln -sf ~/projects/ravan_internship/Moco-Imagenet/imagenet100 ./imagenet100

python main_lincls.py \
    --data ./imagenet100 \
    --pretrained ${CKPT} \
    --arch resnet18 \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160
