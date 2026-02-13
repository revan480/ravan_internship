"""
Collect ResNet-50 MoCo v2 results into a CSV file.

Runs linear evaluation for both experiments and saves results.
Called automatically at the end of run_weekend.sh.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ---- Reuse classes/functions from main_lincls.py ----

class RotationDataset(torch.utils.data.Dataset):
    ANGLES = [0, 90, 180, 270]

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset) * 4

    def __getitem__(self, index):
        img_index = index // 4
        rotation_label = index % 4
        angle = self.ANGLES[rotation_label]
        img_path, _ = self.base_dataset.samples[img_index]
        img = Image.open(img_path).convert("RGB")
        if angle != 0:
            img = img.rotate(angle)
        if self.transform is not None:
            img = self.transform(img)
        return img, rotation_label


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_pretrained_weights(model, pretrained_path):
    print(f"=> Loading pre-trained checkpoint: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if not k.startswith("encoder_q."):
            continue
        new_key = k[len("encoder_q."):]
        if new_key.startswith("fc."):
            skipped.append(new_key)
            continue
        new_state_dict[new_key] = v
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded {len(new_state_dict)} weight tensors")
    print(f"  Skipped projection head keys: {skipped}")
    print(f"  Missing keys (expected): {msg.missing_keys}")
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> Pre-trained weights loaded successfully")


def evaluate_model(data_dir, checkpoint_path, arch="resnet50", eval_rotation=False,
                   epochs=200, lr=30.0, schedule=(120, 160), batch_size=64):
    """Run full linear evaluation and return (acc1, acc5)."""
    num_classes = 4 if eval_rotation else len(
        [d for d in os.listdir(os.path.join(data_dir, "train"))
         if os.path.isdir(os.path.join(data_dir, "train", d))]
    )

    # Build model
    model = models.__dict__[arch](num_classes=num_classes)
    load_pretrained_weights(model, checkpoint_path)

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

    # Transforms
    # ImageNet mean/std — standard normalization values computed across 1.2M ImageNet images
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
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

    traindir = os.path.join(data_dir, "train")
    testdir = os.path.join(data_dir, "test")

    if eval_rotation:
        base_train = datasets.ImageFolder(traindir)
        base_test = datasets.ImageFolder(testdir)
        train_dataset = RotationDataset(base_train, transform=train_transform)
        test_dataset = RotationDataset(base_test, transform=test_transform)
    else:
        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
        test_dataset = datasets.ImageFolder(testdir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    best_acc1 = 0.0
    best_acc5 = 0.0
    topk = (1, min(5, num_classes))

    for epoch in range(epochs):
        # Adjust LR
        cur_lr = lr
        for milestone in schedule:
            if epoch >= milestone:
                cur_lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

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

        # Evaluate
        model.eval()
        top1_sum = 0.0
        top5_sum = 0.0
        total = 0
        with torch.no_grad():
            for images, target in test_loader:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = model(images)
                acc1, acc5 = accuracy(output, target, topk=topk)
                top1_sum += acc1.item() * images.size(0)
                top5_sum += acc5.item() * images.size(0)
                total += images.size(0)

        epoch_acc1 = top1_sum / total
        epoch_acc5 = top5_sum / total

        if epoch_acc1 > best_acc1:
            best_acc1 = epoch_acc1
            best_acc5 = epoch_acc5

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            task = "rotation" if eval_rotation else "species"
            print(f"    [{task}] Epoch {epoch+1}/{epochs}  "
                  f"Acc@1: {epoch_acc1:.2f}%  Best: {best_acc1:.2f}%")

    return best_acc1, best_acc5


def main():
    data_dir = os.path.expanduser("~/Desktop/ravan/moco/cub200_prepared")

    experiments = [
        {
            "name": "exp_color_r50",
            "color": True,
            "rotation": False,
            "checkpoint": "./checkpoints/exp_color_r50/checkpoint_0500.pth.tar",
        },
        {
            "name": "exp_rotation_r50",
            "color": False,
            "rotation": True,
            "checkpoint": "./checkpoints/exp_rotation_r50/checkpoint_0500.pth.tar",
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp['name']}")
        print(f"  Color: {exp['color']}  Rotation: {exp['rotation']}")
        print(f"{'='*60}")

        ckpt = exp["checkpoint"]
        if not os.path.exists(ckpt):
            print(f"  WARNING: Checkpoint not found: {ckpt}")
            print(f"  Skipping this experiment.")
            continue

        # Species eval
        print(f"  >> Bird species (200 classes)...")
        t0 = time.time()
        sp_acc1, sp_acc5 = evaluate_model(data_dir, ckpt, eval_rotation=False)
        print(f"     Done in {time.time()-t0:.0f}s  =>  Acc@1: {sp_acc1:.2f}%  Acc@5: {sp_acc5:.2f}%")

        # Rotation eval
        print(f"  >> Rotation (4 classes)...")
        t0 = time.time()
        rot_acc1, rot_acc5 = evaluate_model(data_dir, ckpt, eval_rotation=True)
        print(f"     Done in {time.time()-t0:.0f}s  =>  Acc@1: {rot_acc1:.2f}%  Acc@5: {rot_acc5:.2f}%")

        results.append({
            "experiment": exp["name"],
            "color_aug": exp["color"],
            "rotation_aug": exp["rotation"],
            "species_top1": sp_acc1,
            "species_top5": sp_acc5,
            "rotation_top1": rot_acc1,
            "rotation_top4": rot_acc5,
        })

    # Write CSV
    csv_path = "results_r50.csv"
    with open(csv_path, "w") as f:
        # Hyperparameters header
        f.write("## =============================================\n")
        f.write("## ResNet-50 MoCo v2 Results (matching paper)\n")
        f.write("## =============================================\n")
        f.write("parameter,ours,paper (Xiao et al. ICLR 2021)\n")
        f.write("backbone,ResNet-50,ResNet-50\n")
        f.write("backbone_params,23.5M,23.5M\n")
        f.write("feature_dim,2048,2048\n")
        f.write("projection_head,MLP (2048-d hidden + ReLU),MLP (2048-d hidden + ReLU)\n")
        f.write("projection_dim,128,128\n")
        f.write("pretrain_epochs,500,500\n")
        f.write("effective_batch_size,288 (48x6 grad accum),256\n")
        f.write("optimizer,SGD,SGD\n")
        f.write("learning_rate,0.03,0.03\n")
        f.write("lr_schedule,step_decay [300 400],step_decay [300 400]\n")
        f.write("moco_queue_size,16384,16384\n")
        f.write("moco_momentum,0.999,0.999\n")
        f.write("moco_temperature,0.2,0.2\n")
        f.write("color_jitter,ColorJitter(0.4 0.4 0.4 0.1) p=0.8,ColorJitter(0.4 0.4 0.4 0.1) p=0.8\n")
        f.write("rotation,RandomRotation90 p=0.5,RandomRotation90 p=0.5\n")
        f.write("eval_epochs,200,200\n")
        f.write("eval_lr,30.0,30.0\n")
        f.write("pretrain_dataset,CUB-200 (~6k images),IN-100 (~125k images)\n")
        f.write("eval_dataset,CUB-200-2011,CUB-200-2011\n")
        f.write("gpu,RTX 3060 (12GB) x1,not specified\n")
        f.write("\n")

        # Our results
        f.write("## =============================================\n")
        f.write("## OUR RESULTS (ResNet-50 MoCo v2 on CUB-200)\n")
        f.write("## =============================================\n")
        f.write("experiment,color_aug,rotation_aug,species_top1,species_top5,rotation_top1,rotation_top4\n")
        for r in results:
            f.write(f"{r['experiment']},{r['color_aug']},{r['rotation_aug']},"
                    f"{r['species_top1']:.2f},{r['species_top5']:.2f},"
                    f"{r['rotation_top1']:.2f},{r['rotation_top4']:.2f}\n")

        f.write("\n")
        f.write("## =============================================\n")
        f.write("## PAPER RESULTS — Table 1: MoCo on IN-100\n")
        f.write("## =============================================\n")
        f.write("model,color_aug,rotation_aug,rotation_acc,IN100_top1,IN100_top5\n")
        f.write("MoCo,True,False,61.1,81.0,95.2\n")
        f.write("MoCo + Rotation,True,True,43.3,79.4,94.1\n")
        f.write("\n")
        f.write("## =============================================\n")
        f.write("## PAPER RESULTS — Table 2: MoCo on CUB-200\n")
        f.write("## =============================================\n")
        f.write("model,color_aug,rotation_aug,CUB200_top1,CUB200_top5\n")
        f.write("MoCo,True,False,36.7,64.7\n")

        # Direct comparison
        if len(results) >= 1:
            f.write("\n")
            f.write("## =============================================\n")
            f.write("## DIRECT COMPARISON\n")
            f.write("## =============================================\n")
            f.write("metric,ours,paper,note\n")
            r_color = next((r for r in results if r["color_aug"]), None)
            r_rot = next((r for r in results if r["rotation_aug"]), None)
            if r_color:
                f.write(f"species_top1 (color aug),{r_color['species_top1']:.2f},36.7,paper pretrained on IN-100 (~125k images)\n")
                f.write(f"species_top5 (color aug),{r_color['species_top5']:.2f},64.7,paper pretrained on IN-100 (~125k images)\n")
                f.write(f"rotation_acc (no rot aug),{r_color['rotation_top1']:.2f},61.1,rotation info preserved\n")
            if r_rot:
                f.write(f"rotation_acc (with rot aug),{r_rot['rotation_top1']:.2f},43.3,rotation info destroyed\n")
            if r_color and r_rot:
                drop = r_color['rotation_top1'] - r_rot['rotation_top1']
                f.write(f"rotation_drop,{drop:.2f},17.80,both should show rotation info destroyed by rotation aug\n")

    print(f"\n=> Results saved to: {csv_path}")

    # Print summary table
    print(f"\nRESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<25s} {'Color':>5s} {'Rot':>5s} | {'Species@1':>10s} {'Species@5':>10s} | {'Rot@1':>8s} {'Rot@4':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment']:<25s} {str(r['color_aug']):>5s} {str(r['rotation_aug']):>5s} | "
              f"{r['species_top1']:>9.2f}% {r['species_top5']:>9.2f}% | "
              f"{r['rotation_top1']:>7.2f}% {r['rotation_top4']:>7.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
