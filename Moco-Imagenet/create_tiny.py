"""
Create a tiny subset of ImageNet-100 for fast debugging.
Selects 5 random classes, copies 50 train + 10 val images per class.
"""

import os
import random
import shutil

SRC = "./imagenet100"
DST = "./imagenet100_tiny"
NUM_CLASSES = 5
TRAIN_PER_CLASS = 50
VAL_PER_CLASS = 10
SEED = 42


def main():
    random.seed(SEED)

    train_src = os.path.join(SRC, "train")
    val_src = os.path.join(SRC, "val")

    all_classes = sorted(os.listdir(train_src))
    selected = random.sample(all_classes, NUM_CLASSES)
    print(f"Selected {NUM_CLASSES} classes: {selected}")

    total_train = 0
    total_val = 0

    for cls in selected:
        # Train
        src_dir = os.path.join(train_src, cls)
        dst_dir = os.path.join(DST, "train", cls)
        os.makedirs(dst_dir, exist_ok=True)
        images = sorted(os.listdir(src_dir))
        sampled = random.sample(images, min(TRAIN_PER_CLASS, len(images)))
        for img in sampled:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        total_train += len(sampled)
        print(f"  {cls}: {len(sampled)} train images")

        # Val
        src_dir = os.path.join(val_src, cls)
        dst_dir = os.path.join(DST, "val", cls)
        os.makedirs(dst_dir, exist_ok=True)
        images = sorted(os.listdir(src_dir))
        sampled = random.sample(images, min(VAL_PER_CLASS, len(images)))
        for img in sampled:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        total_val += len(sampled)
        print(f"  {cls}: {len(sampled)} val images")

    print(f"\nDone! Created {DST}/")
    print(f"  Train: {NUM_CLASSES} classes, {total_train} images")
    print(f"  Val:   {NUM_CLASSES} classes, {total_val} images")


if __name__ == "__main__":
    main()
