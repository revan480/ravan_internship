# relssl — Relational / Pairwise Augmentation-Prediction SSL

A unified, config-driven SSL repo implementing a **relational** auxiliary task: a
3-layer head looks at *both* views' post-avgpool backbone features `(h1, h2)` and
predicts, **per augmentation factor**, whether the *same* parameter value was applied
to both views (binary same/different). Framework-agnostic across SimCLR / MoCo / BYOL
/ LooC. Total loss `L = L_ssl + λ · mean_f BCE_f`, `λ = 0.5`.

This is a standalone sibling of the existing `*-Imagenet*` repos; it does not modify
any of them. It reuses their conventions (ImageNet-100 / CUB-200 / Flowers-102 data
layout, `ts_ssl_gpu` conda env, "Best Val" logging, checkpoint format).

## Layout
```
configs/    base.yaml + framework/{simclr,moco,byol,looc}.yaml + experiment/{baseline,relpred,relpred_lambda0}.yaml
data/       transforms.py (parameterized per-factor sharing + labels + mask), loader.py
models/     backbones.py (resnet + avgpool hook), rel_head.py, frameworks/{simclr,moco,byol,looc}.py
losses.py   NTXentLoss, BYOLLoss, RelPairLoss (per-factor BCE)
train.py    single entrypoint (--framework / --experiment)
eval/       linear_probe.py (IN-100, CUB-200, --eval-rotation), few_shot.py (Flowers-102)
scripts/    run_pipeline.sh, run_pilot.sh, make_pilot_subset.py, check_pilot_gate.py,
            sbatch_pretrain.slurm, sbatch_eval.slurm, launch_matrix.sh, extract_results.py
tests/      pytest suite (transforms, models, frameworks, pilot gate, extract)
```

## The 8 factors
`rotation, hflip, brightness, contrast, saturation, hue, grayscale, blur` (crop is
excluded — it stays independent as the contrastive signal). Per factor: with prob
`p_same=0.5` the identical parameter is applied to both views (label "same"), else a
guaranteed-different value (discrete: exclude view-1's value; continuous: ≥δ gap).
Saturation/hue are masked out of the loss whenever either view is grayscale.

## Experiments
- `baseline`   — standard independent augmentation, no head (`rel_lambda=0`). Reproduces existing numbers.
- `relpred`    — the method (sharing loader + head, `rel_lambda=0.5`).
- `relpred_lambda0` — ablation: sharing loader, head off. (Not a pure-SSL control.)

## Quick start (cluster, `ts_ssl_gpu`)
```bash
# Phase-2 sanity pilot + automated gate (resnet18, IN-100 subset)
GPU=0 bash relssl/scripts/run_pilot.sh

# One full pipeline (pretrain -> IN-100 lincls -> rotation -> CUB-200 -> Flowers few-shot)
GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash relssl/scripts/run_pipeline.sh

# Full matrix via SLURM (edit #SBATCH placeholders first)
ARCH=resnet50 EPOCHS=500 bash relssl/scripts/launch_matrix.sh

# Collect results
python relssl/scripts/extract_results.py --logs-dir ./relssl/logs --out ./relssl/results.csv
```

Pretraining defaults match each framework's original (`main_*.py`): SimCLR/BYOL
`lr 0.3` ×bs/256, cosine; MoCo/LooC `lr 0.03`, step-decay [300,400]; bs 256, 500 epochs.

## Data paths (defaults, relative to repo root)
- ImageNet-100: `./imagenet100` (symlink to `Moco-Imagenet/imagenet100`)
- CUB-200: `../moco/cub200_prepared`
- Flowers-102: `../flowers102_prepared`

## Tests
```bash
python -m pytest relssl/tests/ -q     # CPU-only; no GPU/data needed
```
