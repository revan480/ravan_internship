# datasets/

All datasets for `relssl` live here, **next to the `relssl/` code**. Training and eval
read from these three subfolders (override the paths in relctl's **Runtime** group or
via the `IN100` / `CUB` / `FLOWERS` env vars):

```
datasets/
├── imagenet100/          { train/  val/ }    # pretraining + IN-100 object/rotation eval
├── cub200_prepared/      { train/  val/ }    # CUB-200 linear eval
└── flowers102_prepared/  { train/  test/ }   # Flowers-102 few-shot eval
```

Each is a standard ImageFolder tree: `<split>/<class>/<images>`.

## How to populate it

**From the Google Drive archives** (see `relssl/HANDOFF.md` Section 3):
```bash
# run from the folder that contains relssl/ and this datasets/
tar xf imagenet100.tar         -C datasets
tar xf cub200_prepared.tar     -C datasets
tar xf flowers102_prepared.tar -C datasets
```

**Or symlink existing copies into here:**
```bash
ln -sfn /path/to/imagenet100        datasets/imagenet100
ln -sfn /path/to/cub200_prepared    datasets/cub200_prepared
ln -sfn /path/to/flowers102_prepared datasets/flowers102_prepared
```

Verify:
```bash
for p in datasets/imagenet100/train datasets/cub200_prepared/train datasets/flowers102_prepared/train; do
  [ -d "$p" ] && echo "OK   $p" || echo "MISSING $p"
done
```

> The actual image data is git-ignored; only this README is tracked, so the folder
> structure travels with the repo while the (huge) data does not.
