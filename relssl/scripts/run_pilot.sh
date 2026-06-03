#!/bin/bash
# Phase-2 pilot: short ResNet-18 relpred run on an ImageNet-100 subset, then the
# automated gate check. Non-SLURM (matches the existing nohup/CUDA_VISIBLE_DEVICES
# convention); run on the cluster GPU.
#
# Usage:
#   GPU=0 bash relssl/scripts/run_pilot.sh
#   GPU=0 EPOCHS=5 FRAMEWORK=simclr bash relssl/scripts/run_pilot.sh
set -e

# --- config (override via env) ---
GPU=${GPU:-0}
FRAMEWORK=${FRAMEWORK:-simclr}
EPOCHS=${EPOCHS:-5}
ARCH=${ARCH:-resnet18}
SRC=${SRC:-./imagenet100}                 # full IN-100 (symlinked as in the existing scripts)
SUBSET=${SUBSET:-./relssl/pilot_in100}    # symlinked pilot subset
N_CLASSES=${N_CLASSES:-10}
N_PER_CLASS=${N_PER_CLASS:-200}
SAVE_DIR=${SAVE_DIR:-./relssl/checkpoints/pilot_${FRAMEWORK}}
LOG=${LOG:-./relssl/logs/pilot_${FRAMEWORK}.log}
CONDA_ENV=${CONDA_ENV:-pytorch_2_0_0}

export CUDA_VISIBLE_DEVICES=${GPU}
# cd to repo root (parent of relssl/)
cd "$(dirname "$0")/../.."
mkdir -p "$(dirname "$LOG")"

# conda env (override with CONDA_ENV=...)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" || echo "WARN: could not activate ${CONDA_ENV} (continuing in current env)"
fi

echo "=========================================="
echo "Phase-2 pilot: ${FRAMEWORK} / ${ARCH} / ${EPOCHS} epochs"
echo "Started: $(date)"
echo "=========================================="

# Ensure the repo-root imagenet100 symlink exists (matches the existing repos).
if [ ! -e "${SRC}" ] && [ -d "./Moco-Imagenet/imagenet100" ]; then
    ln -sf "$(pwd)/Moco-Imagenet/imagenet100" "${SRC}"
fi

# Build the symlinked pilot subset (idempotent: existing symlinks are skipped, so a
# partial subset from an interrupted run is simply completed).
python relssl/scripts/make_pilot_subset.py --src "${SRC}" --dst "${SUBSET}" \
    --n-classes "${N_CLASSES}" --n-per-class "${N_PER_CLASS}" --splits train val

# Pilot pretraining (tee to log + console, matching the repo convention).
python -m relssl.train --framework "${FRAMEWORK}" --experiment relpred \
    --arch "${ARCH}" --data "${SUBSET}" --epochs "${EPOCHS}" \
    --batch-size 256 --workers 8 --save-dir "${SAVE_DIR}" 2>&1 | tee "${LOG}"

echo "=========================================="
echo "Pilot gate check"
echo "=========================================="
python relssl/scripts/check_pilot_gate.py "${LOG}"   # exit code gates scaling
echo "Finished: $(date)"
