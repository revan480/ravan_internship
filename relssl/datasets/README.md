# relssl/datasets/

All datasets for `relssl` live here, **inside the `relssl/` folder**, so pulling just
`relssl/` from GitHub gives you the whole project structure. Training and eval read
from these three subfolders (override the paths in relctl's **Runtime** group or via
the `IN100` / `CUB` / `FLOWERS` env vars):

```
relssl/datasets/
├── imagenet100/          { train/  val/ }    # pretraining + IN-100 object/rotation eval
├── cub200_prepared/      { train/  val/ }    # CUB-200 linear eval
└── flowers102_prepared/  { train/  test/ }   # Flowers-102 few-shot eval
```

Each is a standard ImageFolder tree: `<split>/<class>/<images>`.

> Run all commands from the folder that *contains* `relssl/` — paths are written as
> `./relssl/datasets/...`.

## How to populate it

**From the Google Drive archives** (see `relssl/HANDOFF.md` Section 3):
```bash
# from the folder that contains relssl/
tar xf imagenet100.tar         -C relssl/datasets
tar xf cub200_prepared.tar     -C relssl/datasets
tar xf flowers102_prepared.tar -C relssl/datasets
```

**Or symlink existing copies into here:**
```bash
ln -sfn /path/to/imagenet100         relssl/datasets/imagenet100
ln -sfn /path/to/cub200_prepared     relssl/datasets/cub200_prepared
ln -sfn /path/to/flowers102_prepared relssl/datasets/flowers102_prepared
```

Verify:
```bash
for p in relssl/datasets/imagenet100/train relssl/datasets/cub200_prepared/train relssl/datasets/flowers102_prepared/train; do
  [ -d "$p" ] && echo "OK   $p" || echo "MISSING $p"
done
```

> The actual image data is git-ignored; only this README is tracked, so the folder
> structure travels with the repo while the (huge) data does not.
