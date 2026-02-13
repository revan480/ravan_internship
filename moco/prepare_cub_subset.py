"""
Create a small subset of CUB-200-2011 for quick smoke testing.
Uses only 5 classes with up to 10 train + 5 test images each.

Usage:
    python prepare_cub_subset.py
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Create CUB-200 subset for testing")
    parser.add_argument(
        "--cub_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/ravan/moco/CUB_200_2011"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/ravan/moco/cub200_subset"),
    )
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=10, help="max train images per class")
    parser.add_argument("--max-test", type=int, default=5, help="max test images per class")
    args = parser.parse_args()

    cub_dir = args.cub_dir
    output_dir = args.output_dir

    # Read images.txt
    images = {}
    with open(os.path.join(cub_dir, "images.txt")) as f:
        for line in f:
            parts = line.strip().split()
            images[int(parts[0])] = parts[1]

    # Read train_test_split.txt
    splits = {}
    with open(os.path.join(cub_dir, "train_test_split.txt")) as f:
        for line in f:
            parts = line.strip().split()
            splits[int(parts[0])] = int(parts[1])

    # Read image_class_labels.txt
    labels = {}
    with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
        for line in f:
            parts = line.strip().split()
            labels[int(parts[0])] = int(parts[1])

    # Pick first N classes
    selected_classes = set(range(1, args.num_classes + 1))

    # Group by class and split
    train_by_class = {}
    test_by_class = {}
    for img_id, rel_path in sorted(images.items()):
        cls = labels[img_id]
        if cls not in selected_classes:
            continue
        is_train = splits[img_id]
        bucket = train_by_class if is_train else test_by_class
        bucket.setdefault(cls, []).append((img_id, rel_path))

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    train_count = 0
    test_count = 0

    for cls in sorted(selected_classes):
        for split_name, bucket, limit in [
            ("train", train_by_class, args.max_train),
            ("test", test_by_class, args.max_test),
        ]:
            items = bucket.get(cls, [])[:limit]
            for _, rel_path in items:
                class_name = rel_path.split("/")[0]
                filename = rel_path.split("/")[1]
                dst_dir = os.path.join(output_dir, split_name, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                src = os.path.join(cub_dir, "images", rel_path)
                dst = os.path.join(dst_dir, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                if split_name == "train":
                    train_count += 1
                else:
                    test_count += 1

    print(f"Subset created at: {output_dir}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Train:   {train_count} images")
    print(f"  Test:    {test_count} images")


if __name__ == "__main__":
    main()
