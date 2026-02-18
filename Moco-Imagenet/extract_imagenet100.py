"""
Extract ImageNet-100 subset from HuggingFace parquet files.

Uses the Tian et al. 2019 (CMC paper) 100-class split, same as the LooC paper.
Outputs PyTorch ImageFolder structure:
    ./imagenet100/train/<synset_id>/*.JPEG
    ./imagenet100/val/<synset_id>/*.JPEG

Usage:
    python extract_imagenet100.py
"""

import os
import sys

import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 100 synset IDs from Tian et al. 2019 (CMC paper) used by LooC paper
# ---------------------------------------------------------------------------
IN100_SYNSETS = {
    "n02869837", "n01749939", "n02488291", "n02107142", "n13037406",
    "n02091831", "n04517823", "n04589890", "n03062245", "n01773797",
    "n01735189", "n07831146", "n07753275", "n03085013", "n04485082",
    "n02105505", "n01983481", "n02788148", "n03530642", "n04435653",
    "n02086910", "n02859443", "n13040303", "n03594734", "n02085620",
    "n02099849", "n01558993", "n04493381", "n02109047", "n04111531",
    "n02877765", "n04429376", "n02009229", "n01978455", "n02106550",
    "n01820546", "n01692333", "n07714571", "n02974003", "n02114855",
    "n03785016", "n03764736", "n03775546", "n02087046", "n07836838",
    "n04099969", "n04592741", "n03891251", "n02701002", "n03379051",
    "n02259212", "n07715103", "n03947888", "n04026417", "n02326432",
    "n03637318", "n01980166", "n02113799", "n02086240", "n03903868",
    "n02483362", "n04127249", "n02089973", "n03017168", "n02093428",
    "n02804414", "n02396427", "n04418357", "n02172182", "n01729322",
    "n02113978", "n03787032", "n02089867", "n02119022", "n03777754",
    "n04238763", "n02231487", "n03032252", "n02138441", "n02104029",
    "n03837869", "n03494278", "n04136333", "n03794056", "n03492542",
    "n02018207", "n04067472", "n03930630", "n03584829", "n02123045",
    "n04229816", "n02100583", "n03642806", "n04336792", "n03259280",
    "n02116738", "n02108089", "n03424325", "n01855672", "n02090622",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "imagenet-1k", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "imagenet100")

# Add parent so we can import classes.py
sys.path.insert(0, os.path.join(SCRIPT_DIR, "imagenet-1k"))
from classes import IMAGENET2012_CLASSES


def build_label_to_synset():
    """Build mapping from integer label (0-999) to synset ID.

    HuggingFace ImageNet labels 0-999 correspond to synsets sorted
    alphabetically. IMAGENET2012_CLASSES is already an OrderedDict
    in sorted synset order, so index = label.
    """
    synsets = list(IMAGENET2012_CLASSES.keys())
    return {i: s for i, s in enumerate(synsets)}


def get_parquet_files(split):
    """Get sorted list of parquet files for a split ('train' or 'validation')."""
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith(split) and f.endswith(".parquet")
    )
    return [os.path.join(DATA_DIR, f) for f in files]


def extract_split(split_name, parquet_prefix, out_subdir, label_to_synset, target_labels):
    """Extract images for one split (train or val).

    Processes parquet files one at a time to keep memory low.
    For each file, reads only the 'label' column first to find matching rows,
    then reads image data only for those rows.
    """
    parquet_files = get_parquet_files(parquet_prefix)
    if not parquet_files:
        print(f"  No parquet files found for split '{parquet_prefix}'")
        return {}

    # Per-class image counters for unique filenames
    counters = {synset: 0 for synset in IN100_SYNSETS}
    # Per-class total counts for summary
    class_counts = {synset: 0 for synset in IN100_SYNSETS}

    # Create output directories
    for synset in IN100_SYNSETS:
        os.makedirs(os.path.join(out_subdir, synset), exist_ok=True)

    print(f"\n  Processing {len(parquet_files)} parquet files for {split_name}...")

    for pf in tqdm(parquet_files, desc=f"  {split_name}"):
        # Step 1: Read only the label column to find matching row indices
        table = pq.read_table(pf, columns=["label"])
        labels = table.column("label").to_pylist()

        # Find indices of rows belonging to our 100 classes
        matching_indices = [i for i, lab in enumerate(labels) if lab in target_labels]

        if not matching_indices:
            continue

        # Step 2: Read the full file and extract only matching rows
        full_table = pq.read_table(pf, columns=["image", "label"])

        for idx in matching_indices:
            label = full_table.column("label")[idx].as_py()
            image_dict = full_table.column("image")[idx].as_py()
            image_bytes = image_dict["bytes"]

            synset = label_to_synset[label]
            counter = counters[synset]
            filename = f"{synset}_{counter:05d}.JPEG"
            filepath = os.path.join(out_subdir, synset, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            counters[synset] += 1
            class_counts[synset] += 1

        # Free memory
        del full_table

    return class_counts


def main():
    print("=" * 60)
    print("ImageNet-100 Extraction (Tian et al. 2019 / CMC split)")
    print("=" * 60)

    # Build label -> synset mapping
    label_to_synset = build_label_to_synset()

    # Compute target label integers for the 100 synsets
    target_labels = set()
    for label, synset in label_to_synset.items():
        if synset in IN100_SYNSETS:
            target_labels.add(label)

    print(f"  Target classes: {len(target_labels)} (expected 100)")
    assert len(target_labels) == 100, f"Expected 100 classes, got {len(target_labels)}"

    # Print a few example mappings
    print(f"  Example: label 0 -> {label_to_synset[0]}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract train split
    train_dir = os.path.join(OUTPUT_DIR, "train")
    train_counts = extract_split(
        "train", "train", train_dir, label_to_synset, target_labels
    )

    # Extract val split
    val_dir = os.path.join(OUTPUT_DIR, "val")
    val_counts = extract_split(
        "val", "validation", val_dir, label_to_synset, target_labels
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"  Classes:      {len(IN100_SYNSETS)}")
    print(f"  Train images: {total_train}")
    print(f"  Val images:   {total_val}")
    print(f"  Total:        {total_train + total_val}")
    print(f"  Output:       {OUTPUT_DIR}")

    # Per-class breakdown
    print(f"\n  Per-class counts (train / val):")
    for synset in sorted(IN100_SYNSETS):
        t = train_counts.get(synset, 0)
        v = val_counts.get(synset, 0)
        name = IMAGENET2012_CLASSES[synset].split(",")[0]
        print(f"    {synset} ({name:30s}): {t:5d} / {v:3d}")


if __name__ == "__main__":
    main()
