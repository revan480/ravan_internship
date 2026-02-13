#!/bin/bash
# =============================================================================
# MoCo v2 on CUB-200-2011 — Full Experiment Pipeline
#
# Reproduces the LooC paper observation (Xiao et al., ICLR 2021):
# contrastive learning destroys augmentation-specific information.
#
# Hardware: Single NVIDIA RTX 3060 (12GB VRAM)
# =============================================================================

set -e  # Exit on error

CUB_DIR=~/Desktop/ravan/moco/CUB_200_2011
DATA_DIR=~/Desktop/ravan/moco/cub200_prepared
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

echo "============================================================"
echo "Step 1: Prepare CUB-200-2011 dataset"
echo "============================================================"
python prepare_cub.py --cub_dir "$CUB_DIR" --output_dir "$DATA_DIR"
echo ""

echo "============================================================"
echo "Step 2: MoCo v2 Pre-Training (4 experiments)"
echo "============================================================"

echo ""
echo "--- Experiment A: Color augmentation (default MoCo v2) ---"
python main_moco.py \
    --data "$DATA_DIR" \
    --epochs 500 \
    --save-dir ./checkpoints/exp_color

echo ""
echo "--- Experiment B: Rotation augmentation only (no color) ---"
python main_moco.py \
    --data "$DATA_DIR" \
    --epochs 500 \
    --no-color \
    --use-rotation \
    --save-dir ./checkpoints/exp_rotation

echo ""
echo "--- Experiment C: Both color + rotation ---"
python main_moco.py \
    --data "$DATA_DIR" \
    --epochs 500 \
    --use-rotation \
    --save-dir ./checkpoints/exp_color_rotation

echo ""
echo "--- Experiment D: Baseline (no color, no rotation) ---"
python main_moco.py \
    --data "$DATA_DIR" \
    --epochs 500 \
    --no-color \
    --save-dir ./checkpoints/exp_baseline

echo ""
echo "============================================================"
echo "Step 3: Linear Evaluation — Bird Species (200 classes)"
echo "============================================================"

for exp in exp_color exp_rotation exp_color_rotation exp_baseline; do
    echo ""
    echo "--- Evaluating: $exp ---"
    python main_lincls.py \
        --data "$DATA_DIR" \
        --pretrained "./checkpoints/${exp}/checkpoint_0500.pth.tar" \
        --arch resnet18
done

echo ""
echo "============================================================"
echo "Step 4: Linear Evaluation — Rotation (4 classes)"
echo "============================================================"

for exp in exp_color exp_rotation exp_color_rotation exp_baseline; do
    echo ""
    echo "--- Evaluating rotation: $exp ---"
    python main_lincls.py \
        --data "$DATA_DIR" \
        --pretrained "./checkpoints/${exp}/checkpoint_0500.pth.tar" \
        --arch resnet18 \
        --eval-rotation
done

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "============================================================"
