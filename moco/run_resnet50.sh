#!/bin/bash
# =====================================================================
# Weekend Run: ResNet-50 MoCo v2 on CUB-200
#
# Runs 2 pre-training experiments + 4 linear evaluations + saves CSV
#
# Experiment 1: Color=Yes, Rotation=No  (default MoCo augmentation)
# Experiment 2: Color=No,  Rotation=Yes (rotation augmentation only)
#
# Usage: run from moco folder with conda env activated:
#   nohup bash run_weekend.sh > weekend_log.txt 2>&1 &
# =====================================================================

set -e  # stop on error

# Deactivate any venv and activate correct conda env
deactivate 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate ts_ssl_gpu
cd ~/Desktop/ravan/moco

echo "========================================"
echo "Started: $(date)"
echo "========================================"

# ---------------------------------------------------------
# STEP 1: Pre-train MoCo with COLOR augmentation (500 epochs)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 1/6: Pre-training exp_color_r50"
echo "  Color: YES, Rotation: NO"
echo "  Started: $(date)"
echo "========================================"

python main_moco.py \
    --save-dir ./checkpoints/exp_color_r50

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 2: Pre-train MoCo with ROTATION augmentation (500 epochs)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 2/6: Pre-training exp_rotation_r50"
echo "  Color: NO, Rotation: YES"
echo "  Started: $(date)"
echo "========================================"

python main_moco.py \
    --no-color \
    --use-rotation \
    --save-dir ./checkpoints/exp_rotation_r50

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 3: Linear eval — exp_color_r50 → Species (200 classes)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 3/6: Linear eval exp_color_r50 → Species"
echo "  Started: $(date)"
echo "========================================"

python main_lincls.py \
    --pretrained ./checkpoints/exp_color_r50/checkpoint_0500.pth.tar

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 4: Linear eval — exp_color_r50 → Rotation (4 classes)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 4/6: Linear eval exp_color_r50 → Rotation"
echo "  Started: $(date)"
echo "========================================"

python main_lincls.py \
    --pretrained ./checkpoints/exp_color_r50/checkpoint_0500.pth.tar \
    --eval-rotation

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 5: Linear eval — exp_rotation_r50 → Species (200 classes)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 5/6: Linear eval exp_rotation_r50 → Species"
echo "  Started: $(date)"
echo "========================================"

python main_lincls.py \
    --pretrained ./checkpoints/exp_rotation_r50/checkpoint_0500.pth.tar

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 6: Linear eval — exp_rotation_r50 → Rotation (4 classes)
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "STEP 6/6: Linear eval exp_rotation_r50 → Rotation"
echo "  Started: $(date)"
echo "========================================"

python main_lincls.py \
    --pretrained ./checkpoints/exp_rotation_r50/checkpoint_0500.pth.tar \
    --eval-rotation

echo "  Finished: $(date)"

# ---------------------------------------------------------
# STEP 7: Collect results into CSV
# ---------------------------------------------------------
echo ""
echo "========================================"
echo "Collecting results into results_r50.csv"
echo "========================================"

python collect_results_resnet50.py

echo ""
echo "========================================"
echo "ALL DONE! $(date)"
echo "Results saved to: results_r50.csv"
echo "Full log saved to: weekend_log.txt"
echo "========================================"
