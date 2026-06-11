# relssl — Relational / Pairwise Augmentation-Prediction SSL

A unified, config-driven SSL repo implementing a **relational** auxiliary task: a
3-layer head looks at *both* views' post-avgpool backbone features `(h1, h2)` and
predicts, **per augmentation factor**, whether the *same* parameter value was applied
to both views (binary same/different). Framework-agnostic across SimCLR / MoCo / BYOL
/ LooC. Total loss `L = L_ssl + λ · mean_f BCE_f`, `λ = 0.5`.

This is a standalone sibling of the existing `*-Imagenet*` repos; it does not modify
any of them. It reuses their conventions (ImageNet-100 / CUB-200 / Flowers-102 data
layout, `pytorch_2_0_0` conda env, "Best Val" logging, checkpoint format).

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

## Interactive control panel — `relctl` (recommended)

`relctl` is a full interactive TUI for driving everything from one place over SSH —
edit **any** knob, run **any** action, and manage long jobs, without hand-editing
files or remembering flags.

```bash
python -m relssl.relctl            # auto: Rich tier if installed, else plain
python -m relssl.relctl --plain    # force the zero-dependency plain tier
python -m relssl.relctl --validate # check the knob catalog vs configs/, then exit
```

- **Configure** — 7 grouped editors expose every knob with live validation: globals,
  the active framework's block (e.g. SimCLR `temperature`, MoCo `moco_k`, BYOL
  `tau_base`), optimizer/LR (with a live `base_lr` readout), augmentation, the
  relational head (incl. the per-factor `delta` dict and the count-locked
  `rel_factors`), eval/probe hyperparameters, and runtime/paths/pilot.
- **Run** — pilot · pipeline · pretrain-only · eval-only · single eval step · matrix
  (SLURM) · resume · make-subset · gate-check · extract-results · tests. The launch
  screen shows the **resolved** config, the exact command(s), the generated YAML
  overlay, and `run.sh`'s preflight checklist before you commit.
- **Jobs** — launches long runs in the background (`nohup`, or `tmux` if present;
  `sbatch` for matrix) and returns immediately. List/tail/stop, with live epoch +
  per-factor progress scraped from the logs. The registry survives SSH drops.
- **Profiles** — save/load named configurations (tracked in `relctl/profiles/`).

Zero hard dependencies beyond what the `pytorch_2_0_0` env already has (stdlib +
pyyaml). `pip install --user rich` upgrades the rendering automatically. Edited
YAML-only knobs travel to `train.py` via the new `--config-overlay` flag, so the
committed `configs/*.yaml` are never mutated. `python -m relssl.train --print-config`
prints the fully-resolved merged config (what `relctl`'s **verify** uses).

## Static control panel — `relssl/run.sh` (scriptable / non-interactive)

A single bash script to configure, **preview, and run** everything — handy for
scripts and SLURM. Edit the CONFIG block at the top (or override any value with an
env var), preview with `--dry-run` (prints the prerequisite checklist, the resolved
config, and the exact commands — runs nothing), then run it (it asks for confirmation
first, unless `--yes`). `relctl` reuses this script's preflight + dry-run.

```bash
# 1) see what WILL happen, without running anything:
bash relssl/run.sh --dry-run

# 2) run it (confirm prompt):       3) or pick a mode / override knobs:
bash relssl/run.sh                  MODE=pilot GPU=1 EPOCHS=50 bash relssl/run.sh --dry-run
```

Modes: `test` (CPU unit tests only) · `pilot` (short ResNet-18 run + automated gate) ·
`pipeline` (full pretrain→4 evals per framework/experiment) · `matrix` (full SLURM grid).
A real run aborts if any prerequisite check fails; `--dry-run` only reports them.

## Quick start (cluster, `pytorch_2_0_0`)
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

## Data paths (defaults)
All datasets live under a single `datasets/` folder next to `relssl/`:
- ImageNet-100: `./datasets/imagenet100`   (`train/`, `val/`)
- CUB-200: `./datasets/cub200_prepared`     (`train/`, `val/`)
- Flowers-102: `./datasets/flowers102_prepared` (`train/`, `test/`)

Override any of them in relctl's **Runtime** group, or via the `IN100`/`CUB`/`FLOWERS`
env vars for the scripts.

## Tests
```bash
python -m pytest relssl/tests/ -q     # CPU-only; no GPU/data needed
```
