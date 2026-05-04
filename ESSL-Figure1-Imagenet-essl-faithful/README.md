# E-SSL Figure 1 Reproduction on ImageNet-100

Reproduces the logic of E-SSL (Dangovski et al., ICLR 2022) Figure 1, adapted to ImageNet-100 with our augmentation prediction method and three downstream evaluation datasets.

## Experimental Setup

**7 transformations** tested in **2 conditions** = **14 pretraining runs**:

| # | Transformation | Classes | Invariance | Sensitivity |
|---|---------------|---------|------------|-------------|
| 1 | Horizontal flip | 2 | Always flip | p=0.5, predict applied/not |
| 2 | Grayscale | 2 | Always grayscale | p=0.5, predict applied/not |
| 3 | Four-fold rotation | 4 | Always rotate (uniform 0/90/180/270) | Uniform angle, predict which |
| 4 | Vertical flip | 2 | Always vflip | p=0.5, predict applied/not |
| 5 | 2x2 jigsaw | 24 | Always permute (uniform over 4!) | Uniform permutation, predict which |
| 6 | Four-fold blur | 4 | Always blur (uniform kernel 0/5/9/15) | Uniform kernel, predict which |
| 7 | Color inversion | 2 | Always invert | p=0.5, predict applied/not |

**Invariance condition**: Transformation added to SimCLR augmentation with p=1. No prediction head. Standard NT-Xent loss only.

**Sensitivity condition**: Transformation applied stochastically, prediction head on backbone features predicts the label. Loss = NT-Xent + 0.5 * CE(pred, label).

## Differences from Combo-Pred Experiments

Our existing combo-pred work uses a single 4-class head predicting the combination of rotation + color jitter. This experiment isolates each transformation individually, testing 7 different ones with their natural number of classes (2, 4, or 24).

## Architecture

- Backbone: ResNet-18 (512-dim features)
- SimCLR projector: Linear(512, 512) -> ReLU -> Linear(512, 128)
- Prediction head (sensitivity only): Linear(512, num_classes)
- Pretraining: 200 epochs, batch 256, SGD, cosine LR schedule

## How to Run

### Pilot mode (rotation only, quick sanity check)

```bash
GPU=0 bash pilot_mode.sh
```

### Full sweep (all 14 experiments)

```bash
# On GPU 0 (one terminal):
nohup bash launch_all.sh --gpu 0 > launch_gpu0.log 2>&1 &

# Or split across two GPUs manually:
GPU=0 bash pretraining_scripts/run_hflip_inv.sh 2>&1 | tee logs/run_hflip_inv.log
GPU=1 bash pretraining_scripts/run_hflip_sen.sh 2>&1 | tee logs/run_hflip_sen.log
```

### Extract results

```bash
python extract_results.py
# Produces results_summary.csv with all metrics
```

## What Gets Logged Where

- `logs/run_<transform>_<inv|sen>.log` — full pipeline log (pretrain + all evals)
- `checkpoints/<transform>_<invariance|sensitivity>/` — saved model checkpoints

## Evaluation Protocol

1. **ImageNet-100** linear probe: SGD, lr=30, schedule=[120,160], 200 epochs, frozen backbone
2. **CUB-200** linear probe: same protocol
3. **Flowers-102** few-shot: Adam lr=0.03, 250 iterations, 10 trials, 5-shot and 10-shot

## Final Output

The `extract_results.py` script produces a CSV with one row per (transformation, condition) and columns for all downstream metrics. This is the input for bar-chart plots comparing invariance vs. sensitivity per transformation across all three datasets.
