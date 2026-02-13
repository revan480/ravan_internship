"""
Prepare CUB-200-2011 dataset into ImageFolder format for PyTorch.

Reads the official CUB-200-2011 directory structure and reorganizes it into:
    output_dir/
    ├── train/
    │   ├── 001.Black_footed_Albatross/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── ... (200 classes)
    └── test/
        ├── 001.Black_footed_Albatross/
        └── ... (200 classes)

Usage:
    python prepare_cub.py --cub_dir ~/Desktop/ravan/CUB_200_2011 --output_dir ~/Desktop/ravan/cub200_prepared
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 for ImageFolder")
    parser.add_argument(
        "--cub_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/ravan/moco/CUB_200_2011"),
        help="Path to CUB_200_2011 root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/ravan/moco/cub200_prepared"),
        help="Output directory for ImageFolder structure",
    )
    args = parser.parse_args()

    cub_dir = args.cub_dir
    output_dir = args.output_dir

    # Read images.txt: image_id -> relative_path
    images = {}
    with open(os.path.join(cub_dir, "images.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            image_path = parts[1]
            images[image_id] = image_path

    # Read train_test_split.txt: image_id -> is_training (1 = train, 0 = test)
    splits = {}
    with open(os.path.join(cub_dir, "train_test_split.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = int(parts[0])
            is_train = int(parts[1])
            splits[image_id] = is_train

    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    train_count = 0
    test_count = 0

    for image_id, rel_path in sorted(images.items()):
        # rel_path is like "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
        class_name = rel_path.split("/")[0]
        filename = rel_path.split("/")[1]

        split = "train" if splits[image_id] == 1 else "test"
        dst_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        src = os.path.join(cub_dir, "images", rel_path)
        dst = os.path.join(dst_class_dir, filename)

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        if split == "train":
            train_count += 1
        else:
            test_count += 1

    print(f"Dataset prepared at: {output_dir}")
    print(f"  Train images: {train_count}")
    print(f"  Test images:  {test_count}")
    print(f"  Total:        {train_count + test_count}")


if __name__ == "__main__":
    main()
