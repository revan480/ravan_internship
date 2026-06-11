# relssl — Handoff & Reproduction Guide

A complete, self-contained guide to pull the code, place the datasets, and run
**everything** in `relssl` (the relational / pairwise augmentation-prediction SSL
project). Written for running on a Linux GPU/SLURM server.

> TL;DR — clone the repo, drop three dataset folders into the repo root, create the
> conda env, then drive everything from one interactive panel:
> `python -m relssl.relctl`.

---

## 1. What this is

`relssl` is a unified, config-driven self-supervised-learning (SSL) codebase. It
pretrains a ResNet backbone with one of four SSL frameworks (**SimCLR / MoCo / BYOL /
LooC**) plus an optional **relational head** that predicts, per augmentation factor,
whether the same parameter was applied to both views. It then evaluates the frozen
backbone with linear probes (ImageNet-100 object + rotation, CUB-200) and few-shot
(Flowers-102).

Everything is driven through one Python entrypoint (`train.py`) + eval modules, wrapped
by helper scripts, and surfaced through an interactive control panel, **relctl**.
`relssl` is standalone — it imports only itself; it just *reuses the dataset folders*.

---

## 2. Get the code

```bash
# over SSH (needs access to the repo)
git clone git@github.com:revan480/ravan_internship.git
# or over HTTPS
git clone https://github.com/revan480/ravan_internship.git

cd ravan_internship          # <-- this is the REPO ROOT; run all commands from here
git pull                     # if you already cloned, just update
```

The repo root contains `relssl/` plus the original `*-Imagenet` reference repos. **You
only need `relssl/` and the datasets** to run this project. Run every command below
from the **repo root** (the folder that contains `relssl/`), not from inside `relssl/`.

Datasets and checkpoints are **git-ignored** (too large), so they are *not* in the
clone — they are shipped separately (Section 3).

---

## 3. Datasets

Three datasets are needed. Each is a standard `ImageFolder` tree (`split/<class>/*.jpg`).

| Dataset | Used for | Default location (relative to repo root) | Required sub-folders |
|---|---|---|---|
| **ImageNet-100** | pretraining + IN-100 linear/rotation eval | `./imagenet100/` | `train/`, `val/` |
| **CUB-200-2011** | CUB-200 linear eval | `./moco/cub200_prepared/` | `train/`, `test/` |
| **Flowers-102** | few-shot eval | `./flowers102_prepared/` | `train/`, `test/` (`val/` optional) |

Final layout the code expects:

```
ravan_internship/                 # repo root
├── relssl/                       # the code (from git)
├── imagenet100/                  # ImageNet-100
│   ├── train/<synset>/*.JPEG
│   └── val/<synset>/*.JPEG
├── moco/
│   └── cub200_prepared/          # CUB-200
│       ├── train/<class>/*.jpg
│       └── test/<class>/*.jpg
└── flowers102_prepared/          # Flowers-102
    ├── train/<class>/*.jpg
    └── test/<class>/*.jpg
```

### 3a. Packaging for Google Drive (the person who HAS the data)

From the repo root, make one archive per dataset and upload to Google Drive (set each
to **"Anyone with the link"**):

```bash
tar czf imagenet100.tar.gz        imagenet100
tar czf cub200_prepared.tar.gz    -C moco cub200_prepared
tar czf flowers102_prepared.tar.gz flowers102_prepared
```

For each uploaded file, the share link looks like
`https://drive.google.com/file/d/FILE_ID/view?usp=sharing` — copy the **FILE_ID** and
fill it into the table below before sending this document:

| Dataset | Google Drive FILE_ID |
|---|---|
| ImageNet-100 | `__________________` |
| CUB-200 | `__________________` |
| Flowers-102 | `__________________` |

### 3b. Downloading + placing the data (the professor)

```bash
cd ravan_internship           # repo root
pip install gdown             # one-time

# ImageNet-100  ->  ./imagenet100/
gdown <IMAGENET100_FILE_ID> -O imagenet100.tar.gz
tar xzf imagenet100.tar.gz                # creates ./imagenet100/

# CUB-200  ->  ./moco/cub200_prepared/
mkdir -p moco
gdown <CUB200_FILE_ID> -O cub200_prepared.tar.gz
tar xzf cub200_prepared.tar.gz -C moco    # creates ./moco/cub200_prepared/

# Flowers-102  ->  ./flowers102_prepared/
gdown <FLOWERS_FILE_ID> -O flowers102_prepared.tar.gz
tar xzf flowers102_prepared.tar.gz        # creates ./flowers102_prepared/
```

Verify the layout (must print three "OK" lines):

```bash
for p in imagenet100/train moco/cub200_prepared/train flowers102_prepared/train; do
  [ -d "$p" ] && echo "OK   $p" || echo "MISSING $p"
done
```

### 3c. If the data lives elsewhere (e.g. a scratch/data disk)

You do **not** have to place data in the repo root. Point the tools at any location:

- In **relctl**: open group **7) Runtime, paths & pilot** and set `IN100`, `CUB`,
  `FLOWERS` to your paths.
- For the **scripts**: pass them as env vars, e.g.
  `IN100=/data/imagenet100 CUB=/data/cub200_prepared FLOWERS=/data/flowers102_prepared ...`

### 3d. (Optional) Regenerate "prepared" datasets from raw

If you only have the raw datasets, the prep scripts that produced the above are:
`Moco-Imagenet/extract_imagenet100.py`, `moco/prepare_cub.py`,
`flowers102_raw/prepare_flowers102.py`. (Not needed if you use the Drive archives.)

---

## 4. Environment setup

The scripts default to a conda env named `pytorch_2_0_0`. Use it if it exists,
otherwise create an equivalent (any name works — override with `CONDA_ENV=<name>`):

```bash
conda create -n pytorch_2_0_0 python=3.11 -y
conda activate pytorch_2_0_0
pip install torch torchvision numpy pillow pyyaml      # core (required)
pip install rich pytest                                # optional: nicer UI + tests
```

Quick check:

```bash
python -c "import torch, torchvision, yaml; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

(The first `import torch` can take several seconds — that's normal.)

---

## 5. Sanity checks (no GPU/data needed)

```bash
# (a) the control panel's knob catalog matches the configs + argparse
python -m relssl.relctl --validate
#   -> "OK — catalog is in sync ..."

# (b) unit tests (CPU-only; needs pytest)
python -m pytest relssl/tests/ -q

# (c) preview what a run WOULD do, without running it (per mode)
MODE=pilot    bash relssl/run.sh --dry-run
MODE=pipeline bash relssl/run.sh --dry-run
```

`--dry-run` prints a prerequisite checklist (env, GPU, datasets) — a fast way to
confirm the datasets are found before launching anything.

---

## 6. Running everything — the interactive panel (recommended)

```bash
python -m relssl.relctl            # auto: Rich UI if installed, else plain
python -m relssl.relctl --plain    # force the plain (zero-dependency) UI
```

It's a **typed-key menu**: type the letter/number shown, press **Enter**. The top
banner always shows the current plan (`action`, `fw`, `exp`, `arch`, `epochs`,
`base_lr`, how many edits, running jobs).

**Common actions**

| Goal | Keys |
|---|---|
| Change framework (simclr/moco/byol/looc) | `f` → number |
| Change experiment (baseline/relpred/relpred_lambda0) | `e` → number |
| Edit a setting | group number `1`–`7` → row number → new value → `b` |
| Set dataset paths | `7` → edit `IN100` / `CUB` / `FLOWERS` |
| Pick what to run | `a` → number |
| Preview & launch | `r` → `n` (background) / `t` (tmux) / `f` (foreground) / `d` (dry-run) |
| Watch running jobs | `j` → `t <#>` tail, `s <#>` stop |
| See results table | `o` → `x` (extract logs → results.csv) |
| Save / load a config profile | `s` / `l` ; reset edits `z` ; verify resolved config `v` |
| Quit (jobs keep running) | `q` |

In an editor: `*` marks an edited knob, `d` diffs vs defaults, `h <#>` shows help.
Editing group **5** then the `delta` row opens a sub-form for its 5 per-factor keys.

**Example — a quick end-to-end smoke run from the panel:**
1. `a` → `Pilot + gate`  (subset pretrain + automated gate)
2. `7` → set `pilot_epochs`=`2`, `pilot_classes`=`5`, `pilot_per_class`=`50` → `b`
3. `r` → `n`  (launches in the background, returns immediately)
4. `j` → watch progress; `t 1` to tail the log

A full "MoCo baseline, 100 epochs, launch pilot" is just:
`f`→moco, `e`→baseline, `1`→edit `epochs`→`100`→`b`, `a`→pilot, `r`→`n`.

---

## 7. Running everything — non-interactive (scripts & SLURM)

For automation/SLURM you can bypass the UI. All commands run from the repo root.

```bash
# Pilot: short ResNet-18 run on an IN-100 subset + automated gate
GPU=0 bash relssl/scripts/run_pilot.sh

# One full pipeline: pretrain -> IN-100 lincls -> rotation -> CUB-200 -> Flowers few-shot
GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash relssl/scripts/run_pipeline.sh

# Pretrain only / eval only
MODE=pretrain GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash relssl/scripts/run_pipeline.sh
MODE=eval     GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash relssl/scripts/run_pipeline.sh

# Direct entrypoints (full control over every flag)
python -m relssl.train --framework simclr --experiment relpred \
    --arch resnet50 --data ./imagenet100 --epochs 500 \
    --save-dir ./relssl/checkpoints/simclr_relpred
python -m relssl.eval.linear_probe --data ./imagenet100 --arch resnet50 \
    --pretrained ./relssl/checkpoints/simclr_relpred/checkpoint_0500.pth.tar
python -m relssl.eval.linear_probe --data ./imagenet100 --eval-rotation \
    --pretrained <ckpt>
python -m relssl.eval.few_shot --data ./flowers102_prepared --pretrained <ckpt>

# Inspect the fully-resolved config a run will use (no training)
python -m relssl.train --framework moco --experiment baseline --print-config

# Full SLURM matrix (4 frameworks x experiments). FILL the #SBATCH placeholders first:
#   edit relssl/scripts/sbatch_pretrain.slurm + sbatch_eval.slurm  (<PARTITION>, <ACCOUNT>)
ARCH=resnet50 EPOCHS=500 bash relssl/scripts/launch_matrix.sh

# Collect all results into a CSV
python relssl/scripts/extract_results.py --logs-dir ./relssl/logs --out ./relssl/results.csv
```

Useful env-var overrides for any script: `GPU`, `CONDA_ENV`, `ARCH`, `EPOCHS`,
`EVAL_EPOCHS`, `IN100`, `CUB`, `FLOWERS`, `SAVE_DIR`, `CKPT`.

### The three experiments
- `baseline` — standard independent augmentation, no relational head (reproduces the
  existing per-framework numbers).
- `relpred` — the method: per-factor sharing loader + relational head (`rel_lambda=0.5`).
- `relpred_lambda0` — ablation: sharing loader, head off.

---

## 8. Outputs

| What | Where |
|---|---|
| Checkpoints | `relssl/checkpoints/<framework>_<experiment>/checkpoint_<epoch>.pth.tar` |
| Run logs | `relssl/logs/<framework>_<experiment>.log` |
| Collected results | `relssl/results.csv` |
| relctl runtime state | `relssl/.relctl/` (jobs registry, generated config overlays) |
| Saved relctl profiles | `relssl/relctl/profiles/` |

---

## 9. Notes & troubleshooting

- **Run from the repo root.** `python -m relssl.relctl` (and the `bash relssl/...`
  scripts) must be launched from the folder that contains `relssl/`, so Python can find
  the `relssl` package. Running from *inside* `relssl/` gives
  `No module named 'relssl'`.
- **Datasets not found.** Check `--dry-run` output; either place the folders at the
  default paths (Section 3) or override `IN100`/`CUB`/`FLOWERS`.
- **SLURM matrix.** Edit the `#SBATCH` `<PARTITION>` / `<ACCOUNT>` placeholders in
  `relssl/scripts/sbatch_pretrain.slurm` and `sbatch_eval.slurm` before
  `launch_matrix.sh`.
- **GPU select.** `GPU=<index>` (sets `CUDA_VISIBLE_DEVICES`).
- **Long runs.** Pretraining is hundreds of epochs; launch in the background from
  relctl (`r` → `n`/`t`) or via SLURM. relctl's job registry survives an SSH drop.
- **Pretraining defaults** match each framework's original: SimCLR/BYOL `lr 0.3`
  (×bs/256, cosine); MoCo/LooC `lr 0.03`, step-decay `[300,400]`; batch 256, 500 epochs,
  ResNet-50.

---

*For panel internals and the full knob/action catalog, see `relssl/README.md`.*
