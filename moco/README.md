# MoCo v2 on CUB-200-2011

Self-supervised contrastive pre-training using MoCo v2 on the CUB-200-2011 bird dataset.
Designed for single-GPU training (NVIDIA RTX 3060, 12GB VRAM).

This project reproduces the observation from the LooC paper (Xiao et al., ICLR 2021):
contrastive learning makes representations invariant to the augmentations used during
training, effectively destroying augmentation-specific information.

## Setup

```bash
# Create conda environment
conda create -n moco python=3.10 -y
conda activate moco

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# No other dependencies needed — everything uses PyTorch and torchvision
```

## Dataset

Download CUB-200-2011 from the [official Caltech source](https://www.vision.caltech.edu/datasets/cub_200_2011/)
and extract it to `~/Desktop/ravan/CUB_200_2011/`.

Then prepare it into ImageFolder format:

```bash
python prepare_cub.py \
    --cub_dir ~/Desktop/ravan/CUB_200_2011 \
    --output_dir ~/Desktop/ravan/cub200_prepared
```

This creates `train/` and `test/` directories with 200 class subdirectories each.

## Pre-Training

Train MoCo v2 with different augmentation configurations:

```bash
# Experiment A: Color augmentation (default MoCo v2)
python main_moco.py --data ~/Desktop/ravan/cub200_prepared \
    --save-dir ./checkpoints/exp_color

# Experiment B: Rotation only (no color)
python main_moco.py --data ~/Desktop/ravan/cub200_prepared \
    --no-color --use-rotation \
    --save-dir ./checkpoints/exp_rotation

# Experiment C: Both color + rotation
python main_moco.py --data ~/Desktop/ravan/cub200_prepared \
    --use-rotation \
    --save-dir ./checkpoints/exp_color_rotation

# Experiment D: Baseline (crop + flip + blur only)
python main_moco.py --data ~/Desktop/ravan/cub200_prepared \
    --no-color \
    --save-dir ./checkpoints/exp_baseline
```

Key arguments:
- `--arch`: `resnet18` (default, fast) or `resnet50`
- `--epochs`: default 200
- `--batch-size`: default 64 (safe for 12GB VRAM)
- `--moco-k`: queue size, default 4096
- `--use-color` / `--no-color`: toggle color jittering
- `--use-rotation`: enable random {0, 90, 180, 270} degree rotation
- `--color-strength`: multiplier for color jitter intensity

## Linear Evaluation

Freeze the pre-trained backbone and train a linear classifier:

```bash
# Bird species classification (200 classes)
python main_lincls.py --data ~/Desktop/ravan/cub200_prepared \
    --pretrained ./checkpoints/exp_color/checkpoint_0200.pth.tar

# Rotation classification (4 classes)
python main_lincls.py --data ~/Desktop/ravan/cub200_prepared \
    --pretrained ./checkpoints/exp_color/checkpoint_0200.pth.tar \
    --eval-rotation
```

## Run All Experiments

To run the complete pipeline (prepare data, pre-train 4 models, evaluate all):

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## Expected Results

The LooC paper predicts:
- **Color-trained model**: good at bird classification, poor at rotation classification
  (color invariance was learned, rotation info preserved by default)
- **Rotation-trained model**: poor at bird classification, poor at rotation classification
  (rotation info destroyed by invariance)
- **Both augmentations**: destroys both types of information
- **Baseline**: moderate performance on both (no specific invariance enforced)

## Project Structure

```
moco/
├── moco/
│   ├── __init__.py          # Package init
│   ├── builder.py           # MoCo v2 model (encoders + queue)
│   └── loader.py            # TwoCropsTransform + GaussianBlur
├── prepare_cub.py           # CUB-200 → ImageFolder conversion
├── main_moco.py             # MoCo v2 pre-training (single GPU)
├── main_lincls.py           # Linear evaluation (species + rotation)
├── run_experiments.sh       # Full experiment pipeline
└── README.md                # This file
```

## References

- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
- Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv 2020
- Xiao et al., "What Should Not Be Contrastive in Contrastive Learning", ICLR 2021 (LooC)
