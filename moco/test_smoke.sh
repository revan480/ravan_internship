#!/bin/bash
# =============================================================================
# Quick smoke test — runs 2 epochs on a 5-class subset to verify everything works
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SUBSET_DIR=./cub200_subset
CKPT_DIR=./checkpoints/smoke_test

echo "============================================================"
echo "Step 1: Create small subset (5 classes)"
echo "============================================================"
python prepare_cub_subset.py --output_dir "$SUBSET_DIR"
echo ""

echo "============================================================"
echo "Step 2: MoCo v2 pre-train (2 epochs, batch 16)"
echo "============================================================"
python main_moco.py \
    --data "$SUBSET_DIR" \
    --epochs 2 \
    --batch-size 16 \
    --moco-k 32 \
    --save-dir "$CKPT_DIR" \
    --save-freq 2 \
    --schedule 999 \
    --print-freq 1 \
    --workers 2
echo ""

echo "============================================================"
echo "Step 3: Linear eval — species (5 classes)"
echo "============================================================"
python main_lincls.py \
    --data "$SUBSET_DIR" \
    --pretrained "${CKPT_DIR}/checkpoint_0002.pth.tar" \
    --epochs 2 \
    --lr 1.0 \
    --schedule 999 \
    --batch-size 16 \
    --print-freq 1 \
    --workers 2
echo ""

echo "============================================================"
echo "Step 4: Linear eval — rotation (4 classes)"
echo "============================================================"
python main_lincls.py \
    --data "$SUBSET_DIR" \
    --pretrained "${CKPT_DIR}/checkpoint_0002.pth.tar" \
    --eval-rotation \
    --epochs 2 \
    --lr 1.0 \
    --schedule 999 \
    --batch-size 16 \
    --print-freq 1 \
    --workers 2
echo ""

echo "============================================================"
echo "Smoke test PASSED — all scripts work correctly!"
echo "============================================================"

# Cleanup
rm -rf "$SUBSET_DIR" "$CKPT_DIR"
echo "(Cleaned up subset and test checkpoints)"
