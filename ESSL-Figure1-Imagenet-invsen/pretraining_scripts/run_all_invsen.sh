#!/bin/bash
# E-SSL Figure 1: all 7 transformations x invariance_sensitivity
set -e
export CUDA_VISIBLE_DEVICES=${GPU:-0}

cd "$(dirname "$0")/.."
mkdir -p logs

SCRIPTS=(
    run_hflip_invsen.sh
    run_vflip_invsen.sh
    run_rotation_invsen.sh
    run_grayscale_invsen.sh
    run_jigsaw_invsen.sh
    run_blur_invsen.sh
    run_invert_invsen.sh
)

echo "=== Launching all 7 invariance_sensitivity experiments ==="
echo "Started: $(date)"

for script in "${SCRIPTS[@]}"; do
    logfile="logs/${script%.sh}.log"
    echo "--- Running ${script} -> ${logfile} ---"
    GPU=${GPU:-0} bash "pretraining_scripts/${script}" 2>&1 | tee "${logfile}"
done

echo "=== ALL 7 INVARIANCE_SENSITIVITY EXPERIMENTS COMPLETE ==="
echo "Finished: $(date)"
