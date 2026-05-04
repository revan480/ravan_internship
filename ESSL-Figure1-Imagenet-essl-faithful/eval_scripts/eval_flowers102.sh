#!/bin/bash
# Few-shot evaluation on Flowers-102
# Usage: bash eval_scripts/eval_flowers102.sh <checkpoint_path>
set -e
CKPT=${1:?Usage: eval_flowers102.sh <checkpoint_path>}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."

python main_fewshot.py \
    --data ../flowers102_prepared \
    --pretrained ${CKPT} \
    --arch resnet18
