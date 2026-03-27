"""
Prepare Oxford Flowers-102 dataset into ImageFolder format.

Reads raw files (102flowers.tgz, imagelabels.mat, setid.mat) and creates:
    ../flowers102_prepared/
    ├── train/  (1020 images, 10 per class)
    ├── val/    (1020 images, 10 per class)
    └── test/   (6149 images)

Each split has 102 class folders named 001-102.

Usage:
    cd ~/Desktop/ravan/flowers102_raw
    python prepare_flowers102.py
"""

import os
import shutil
import tarfile

import scipy.io


RAW_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(RAW_DIR), "flowers102_prepared")
TGZ_PATH = os.path.join(RAW_DIR, "102flowers.tgz")
LABELS_PATH = os.path.join(RAW_DIR, "imagelabels.mat")
SETID_PATH = os.path.join(RAW_DIR, "setid.mat")


def main():
    # Step 1: Extract images if not already done
    jpg_dir = os.path.join(RAW_DIR, "jpg")
    if not os.path.isdir(jpg_dir):
        print(f"Extracting {TGZ_PATH} ...")
        with tarfile.open(TGZ_PATH, "r:gz") as tar:
            tar.extractall(RAW_DIR)
        print(f"  Extracted to {jpg_dir}")
    else:
        print(f"Images already extracted at {jpg_dir}")

    # Step 2: Load labels (1-indexed, values 1-102)
    labels_mat = scipy.io.loadmat(LABELS_PATH)
    labels = labels_mat["labels"].squeeze()  # shape (8189,)
    print(f"Loaded labels: {labels.shape[0]} images, classes {labels.min()}-{labels.max()}")

    # Step 3: Load split indices (1-indexed)
    setid_mat = scipy.io.loadmat(SETID_PATH)
    trnid = setid_mat["trnid"].squeeze() - 1   # convert to 0-indexed
    valid = setid_mat["valid"].squeeze() - 1
    tstid = setid_mat["tstid"].squeeze() - 1

    splits = {"train": trnid, "val": valid, "test": tstid}
    print(f"Split sizes: train={len(trnid)}, val={len(valid)}, test={len(tstid)}")

    # Step 4: Create output directory structure and copy images
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory already exists: {OUTPUT_DIR}")
        print("  Removing and recreating...")
        shutil.rmtree(OUTPUT_DIR)

    for split_name, indices in splits.items():
        print(f"\nProcessing {split_name} split ({len(indices)} images)...")
        classes_seen = set()

        for idx in indices:
            # Image filename: image_00001.jpg (1-indexed)
            img_name = f"image_{idx + 1:05d}.jpg"
            src = os.path.join(jpg_dir, img_name)

            # Class label (1-102) → folder name (001-102)
            cls = int(labels[idx])
            cls_folder = f"{cls:03d}"
            classes_seen.add(cls_folder)

            dst_dir = os.path.join(OUTPUT_DIR, split_name, cls_folder)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, img_name))

        print(f"  {split_name}: {len(indices)} images across {len(classes_seen)} classes")

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        n_classes = len(os.listdir(split_dir))
        n_images = sum(
            len(files)
            for _, _, files in os.walk(split_dir)
        )
        print(f"  {split_name:5s}: {n_images:5d} images, {n_classes} classes")
    print("=" * 60)


if __name__ == "__main__":
    main()
