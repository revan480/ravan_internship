#!/bin/bash
# Launch all 14 E-SSL experiments sequentially on one GPU
# Usage: bash launch_all.sh --gpu 0
#        bash launch_all.sh --gpu 1
# Run two instances (one per GPU) for parallelism across the two A6000s.
set -e

GPU=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPTS=(
    run_hflip_inv.sh
    run_hflip_sen.sh
    run_grayscale_inv.sh
    run_grayscale_sen.sh
    run_rotation_inv.sh
    run_rotation_sen.sh
    run_vflip_inv.sh
    run_vflip_sen.sh
    run_jigsaw_inv.sh
    run_jigsaw_sen.sh
    run_blur_inv.sh
    run_blur_sen.sh
    run_invert_inv.sh
    run_invert_sen.sh
)

mkdir -p logs

echo "=== Launching all 14 experiments on GPU ${GPU} ==="
echo "Started: $(date)"
echo ""

for script in "${SCRIPTS[@]}"; do
    logfile="logs/${script%.sh}.log"
    echo "--- Running ${script} → ${logfile} ---"
    GPU=${GPU} bash "pretraining_scripts/${script}" 2>&1 | tee "${logfile}"
    echo ""
done

echo "=== ALL 14 EXPERIMENTS COMPLETE ==="
echo "Finished: $(date)"
