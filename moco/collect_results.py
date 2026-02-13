"""
Collect all linear evaluation results into a CSV file.

Usage (from activated conda env):
    conda activate ts_ssl_gpu
    cd ~/Desktop/ravan/moco
    python collect_results.py
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Reuse components from main_lincls
from main_lincls import (
    RotationDataset,
    accuracy,
    load_pretrained_weights,
)


def evaluate_model(data_dir, pretrained_path, arch, eval_rotation, epochs=200, lr=30.0, schedule=None, batch_size=64):
    """Run full linear evaluation and return results dict."""
    if schedule is None:
        schedule = [120, 160]

    cudnn.benchmark = True

    # Detect classes
    traindir = os.path.join(data_dir, "train")
    testdir = os.path.join(data_dir, "test")
    species_classes = len([d for d in os.listdir(traindir) if os.path.isdir(os.path.join(traindir, d))])
    num_classes = 4 if eval_rotation else species_classes
    task = "rotation" if eval_rotation else "species"

    # Build model
    model = models.__dict__[arch](num_classes=num_classes)
    load_pretrained_weights(model, pretrained_path, argparse.Namespace())

    feature_dim = model.fc.in_features
    model.fc = nn.Linear(feature_dim, num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, lr, momentum=0.9, weight_decay=0.0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if eval_rotation:
        base_train = datasets.ImageFolder(traindir)
        base_test = datasets.ImageFolder(testdir)
        train_dataset = RotationDataset(base_train, transform=train_transform)
        test_dataset = RotationDataset(base_test, transform=test_transform)
    else:
        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
        test_dataset = datasets.ImageFolder(testdir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    topk = (1, min(5, num_classes))
    best_acc1 = 0.0
    best_acc5 = 0.0

    for epoch in range(epochs):
        # Adjust LR
        cur_lr = lr
        for milestone in schedule:
            if epoch >= milestone:
                cur_lr *= 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Train
        model.train()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        for images, target in train_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            for images, target in test_loader:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = model(images)
                acc1, acc5 = accuracy(output, target, topk=topk)
                batch_size_cur = target.size(0)
                correct1 += acc1.item() * batch_size_cur / 100.0
                correct5 += acc5.item() * batch_size_cur / 100.0
                total += batch_size_cur

        test_acc1 = 100.0 * correct1 / total
        test_acc5 = 100.0 * correct5 / total

        if test_acc1 > best_acc1:
            best_acc1 = test_acc1
            best_acc5 = test_acc5

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            print(f"    [{task}] Epoch {epoch+1}/{epochs}  Acc@1: {test_acc1:.2f}%  Best: {best_acc1:.2f}%")

    return {"best_acc1": round(best_acc1, 2), "best_acc5": round(best_acc5, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/ravan/moco/cub200_prepared"))
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--output", default="./results.csv")
    args = parser.parse_args()

    experiments = {
        "exp_color":          {"color": True,  "rotation": False},
        "exp_rotation":       {"color": False, "rotation": True},
        "exp_color_rotation": {"color": True,  "rotation": True},
        "exp_baseline":       {"color": False, "rotation": False},
    }

    results = []

    for exp_name, aug_config in experiments.items():
        ckpt = os.path.join(args.ckpt_dir, exp_name, "checkpoint_0500.pth.tar")
        if not os.path.exists(ckpt):
            print(f"  SKIP {exp_name}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"  Color: {aug_config['color']}  Rotation: {aug_config['rotation']}")
        print(f"{'='*60}")

        # Species classification
        print(f"  >> Bird species (200 classes)...")
        t0 = time.time()
        species = evaluate_model(args.data, ckpt, args.arch, eval_rotation=False)
        print(f"     Done in {time.time()-t0:.0f}s  =>  Acc@1: {species['best_acc1']}%  Acc@5: {species['best_acc5']}%")

        # Rotation classification
        print(f"  >> Rotation (4 classes)...")
        t0 = time.time()
        rotation = evaluate_model(args.data, ckpt, args.arch, eval_rotation=True)
        print(f"     Done in {time.time()-t0:.0f}s  =>  Acc@1: {rotation['best_acc1']}%  Acc@5: {rotation['best_acc5']}%")

        results.append({
            "experiment": exp_name,
            "color_aug": aug_config["color"],
            "rotation_aug": aug_config["rotation"],
            "species_acc1": species["best_acc1"],
            "species_acc5": species["best_acc5"],
            "rotation_acc1": rotation["best_acc1"],
            "rotation_acc4": rotation["best_acc5"],
        })

    # Write CSV
    output_path = args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "color_aug", "rotation_aug",
            "species_acc1", "species_acc5",
            "rotation_acc1", "rotation_acc4",
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<25} {'Color':>6} {'Rot':>5} | {'Species@1':>10} {'Species@5':>10} | {'Rot@1':>8} {'Rot@4':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment']:<25} {str(r['color_aug']):>6} {str(r['rotation_aug']):>5} | "
              f"{r['species_acc1']:>9.2f}% {r['species_acc5']:>9.2f}% | "
              f"{r['rotation_acc1']:>7.2f}% {r['rotation_acc4']:>7.2f}%")

    print(f"\n{'='*80}")
    print("LooC Paper Reference (Table 1 — MoCo on IN-100):")
    print(f"  MoCo (color only):      Rotation Acc: 61.1%   IN-100 top-1: 81.0%")
    print(f"  MoCo + Rotation:        Rotation Acc: 43.3%   IN-100 top-1: 79.4%")
    print(f"{'='*80}")

    print(f"\nResults saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
