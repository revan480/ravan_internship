#!/bin/bash
# Linear probe on CUB-200
# Usage: bash eval_scripts/eval_cub200.sh <checkpoint_path>
set -e
CKPT=${1:?Usage: eval_cub200.sh <checkpoint_path>}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."

python main_lincls.py \
    --data ../moco/cub200_prepared \
    --pretrained ${CKPT} \
    --arch resnet18 \
    --batch-size 256 --epochs 200 --lr 30.0 --schedule 120 160
