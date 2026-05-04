"""
Few-shot classification evaluation on Flowers-102.

Protocol (matches LooC paper):
  - Extract features once using frozen backbone
  - For each K-shot value: run N trials
  - Each trial: sample K images/class, train linear classifier with Adam
  - lr=0.03, 250 iterations, report mean +/- 95% CI

Usage:
    python main_fewshot.py \
        --data ../flowers102_prepared \
        --pretrained ./checkpoints/exp_simclr_color_rotation/checkpoint_0500.pth.tar \
        --n-shots 5 10
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot evaluation (SimCLR)")
    parser.add_argument("--data", type=str, required=True,
                        help="path to prepared Flowers-102 dataset")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="path to SimCLR pre-trained checkpoint")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--n-shots", type=int, nargs="+", default=[5, 10],
                        help="number of shots to evaluate (default: 5 10)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="number of random trials (default: 10)")
    parser.add_argument("--lr", type=float, default=0.03,
                        help="learning rate for Adam optimizer (default: 0.03)")
    parser.add_argument("--iterations", type=int, default=250,
                        help="training iterations (default: 250)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for feature extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="base random seed")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_pretrained_weights(model, pretrained_path):
    """Load SimCLR pre-trained weights into the model backbone.

    SimCLR checkpoint keys: backbone.layer1.0.conv1.weight, projector.0.weight, etc.
    Strips 'backbone.' prefix and skips projector / fc keys.
    """
    print(f"=> Loading SimCLR checkpoint: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            skipped.append(k)
            continue
        new_key = k[len("backbone."):]
        if new_key.startswith("fc."):
            skipped.append(k)
            continue
        new_state_dict[new_key] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded {len(new_state_dict)} weight tensors")
    print(f"  Skipped keys: {skipped}")
    print(f"  Missing keys (expected): {msg.missing_keys}")
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, (
        f"Unexpected missing keys: {msg.missing_keys}"
    )
    print("=> Pre-trained weights loaded successfully")


def extract_features(model, dataloader):
    """Extract features from all images using frozen backbone."""
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            features = model(images.cuda())
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def few_shot_trial(train_features, train_labels, test_features, test_labels,
                   k, feat_dim, num_classes, lr, iterations, seed):
    """Run a single few-shot trial: sample K/class, train linear, evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample K images per class
    shot_indices = []
    for cls in range(num_classes):
        cls_indices = (train_labels == cls).nonzero(as_tuple=False).squeeze(1)
        perm = torch.randperm(len(cls_indices))[:k]
        shot_indices.append(cls_indices[perm])
    shot_indices = torch.cat(shot_indices)

    shot_features = train_features[shot_indices]
    shot_labels = train_labels[shot_indices]

    # Train linear classifier
    classifier = nn.Linear(feat_dim, num_classes).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(iterations):
        idx = torch.randint(0, len(shot_features), (min(64, len(shot_features)),))
        batch_feat = shot_features[idx].cuda()
        batch_label = shot_labels[idx].cuda()

        logits = classifier(batch_feat)
        loss = criterion(logits, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_features.cuda())
        pred = test_logits.argmax(dim=1)
        acc = (pred == test_labels.cuda()).float().mean().item() * 100

    return acc


def main():
    args = parse_args()

    # Data transforms (same as linear eval test)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, "train"), transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data, "test"), transform)

    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Build model and load weights
    print(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch]()
    load_pretrained_weights(model, args.pretrained)
    model.fc = nn.Identity()
    model.cuda()
    model.eval()

    feat_dim = 2048 if args.arch == "resnet50" else 512

    # Extract features once
    print("=> Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader)
    print(f"  Train features: {train_features.shape}")

    print("=> Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader)
    print(f"  Test features: {test_features.shape}")

    # Few-shot evaluation
    print("\n" + "=" * 70)
    print("Few-Shot Classification Results — Flowers-102")
    print("=" * 70)
    print(f"  Checkpoint: {args.pretrained}")
    print(f"  Trials: {args.n_trials}")
    print()

    for k in args.n_shots:
        accs = []
        for trial in range(args.n_trials):
            seed = args.seed + trial
            acc = few_shot_trial(
                train_features, train_labels,
                test_features, test_labels,
                k, feat_dim, num_classes, args.lr, args.iterations, seed)
            accs.append(acc)

        mean = np.mean(accs)
        std = np.std(accs)
        ci95 = 1.96 * std / np.sqrt(len(accs))
        print(f"  {k}-shot: {mean:.1f}% (± {ci95:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
