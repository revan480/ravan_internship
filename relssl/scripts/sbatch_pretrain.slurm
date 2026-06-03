#!/bin/bash
#SBATCH --job-name=relssl_pretrain
#SBATCH --partition=<PARTITION>          # TODO: fill in your cluster partition
#SBATCH --account=<ACCOUNT>              # TODO: fill in your account (or remove)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=48:00:00                  # TODO: adjust
#SBATCH --output=relssl/logs/%x_%j.out
#SBATCH --error=relssl/logs/%x_%j.err
#
# Pretrain one (framework, experiment). Submit with env vars, e.g.:
#   sbatch --export=ALL,FRAMEWORK=simclr,EXPERIMENT=relpred,ARCH=resnet50,EPOCHS=500 \
#          relssl/scripts/sbatch_pretrain.slurm
set -e

FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}
IN100=${IN100:-./imagenet100}

# repo root = parent of the relssl/ package
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ts_ssl_gpu

# Ensure the ImageNet-100 symlink exists (matches the existing repos' convention).
[ -e "${IN100}" ] || ln -sf "$(pwd)/Moco-Imagenet/imagenet100" "${IN100}"

MODE=pretrain FRAMEWORK="${FRAMEWORK}" EXPERIMENT="${EXPERIMENT}" ARCH="${ARCH}" \
    EPOCHS="${EPOCHS}" IN100="${IN100}" bash relssl/scripts/run_pipeline.sh
