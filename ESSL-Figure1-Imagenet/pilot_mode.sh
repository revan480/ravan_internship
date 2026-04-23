#!/bin/bash
# Pilot mode: rotation only (invariance + sensitivity)
# Quick sanity check before committing to all 14 runs
# Usage: GPU=0 bash pilot_mode.sh
set -e

echo "=== PILOT MODE: rotation invariance + sensitivity ==="
echo "Started: $(date)"

GPU=${GPU:-0} bash pretraining_scripts/run_rotation_inv.sh 2>&1 | tee logs/run_rotation_inv.log
GPU=${GPU:-0} bash pretraining_scripts/run_rotation_sen.sh 2>&1 | tee logs/run_rotation_sen.log

echo "=== PILOT MODE COMPLETE ==="
echo "Finished: $(date)"
echo "Check logs/run_rotation_*.log for results"
