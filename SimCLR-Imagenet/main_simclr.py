import argparse
import math
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from simclr.builder import SimCLR
from simclr.loader import GaussianBlur, SimCLRTransform
from simclr.loss import NTXentLoss


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Pre-Training (Single GPU)")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="./imagenet100",
        help="path to dataset (with train/ subdirectory)",
    )

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="model architecture (default: resnet50)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=0.3, help="base learning rate (scaled by batch_size/256)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--workers", type=int, default=4, help="data loading workers")

    # SimCLR
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature")
    parser.add_argument("--simclr-dim", type=int, default=128, help="projection output dimension")

    # LR schedule
    parser.add_argument("--cos", action="store_true", default=True, help="use cosine LR schedule (default: True)")
    parser.add_argument("--no-cos", action="store_false", dest="cos", help="disable cosine LR, use step decay")
    parser.add_argument("--schedule", nargs="+", type=int, default=[300, 400], help="LR drop epochs (step decay)")

    # Augmentation flags
    parser.add_argument("--use-color", action="store_true", default=True, dest="use_color", help="enable color jittering (default: True)")
    parser.add_argument("--no-color", action="store_false", dest="use_color", help="disable color jittering")
    parser.add_argument("--use-rotation", action="store_true", default=False, help="enable random 90-degree rotation augmentation")
    parser.add_argument("--color-strength", type=float, default=1.0, help="multiplier for color jittering strength")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./checkpoints/default", help="directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")
    parser.add_argument("--save-freq", type=int, default=50, help="save checkpoint every N epochs")

    # Misc
    parser.add_argument("--print-freq", type=int, default=20, help="print frequency (batches)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    # Scale learning rate: lr = base_lr * batch_size / 256
    args.lr = args.lr * args.batch_size / 256

    return args


class RandomRotation90:
    """Apply a random rotation from {0, 90, 180, 270} degrees."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = img.rotate(angle)
        return img


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays training progress."""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def build_augmentation(args):
    """Build the augmentation pipeline based on experiment flags."""
    aug_list = []

    if args.use_rotation:
        aug_list.append(transforms.RandomApply([RandomRotation90()], p=0.5))

    aug_list.extend([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])

    if args.use_color:
        s = args.color_strength
        aug_list.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                p=0.8,
            )
        )
        aug_list.append(transforms.RandomGrayscale(p=0.2))

    aug_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
    aug_list.append(transforms.ToTensor())
    aug_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(aug_list)


def adjust_learning_rate(optimizer, epoch, args):
    """LR schedule: cosine annealing or step decay."""
    if args.cos:
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        lr = args.lr
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"  => Saved checkpoint: {filepath}")


def main():
    args = parse_args()

    # Print experiment configuration
    print("=" * 70)
    print("SimCLR Pre-Training Configuration")
    print("=" * 70)
    print(f"  Architecture:     {args.arch}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr:.4f} (base 0.3 * {args.batch_size}/256)")
    print(f"  LR schedule:      {'cosine' if args.cos else f'step decay at {args.schedule}'}")
    print(f"  Weight decay:     {args.weight_decay}")
    print(f"  SimCLR dim:       {args.simclr_dim}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Color augment:    {args.use_color}" + (f" (strength={args.color_strength})" if args.use_color else ""))
    print(f"  Rotation augment: {args.use_rotation}" + (" (p=0.5)" if args.use_rotation else ""))
    print(f"  Data:             {args.data}")
    print(f"  Save dir:         {args.save_dir}")
    print(f"  Seed:             {args.seed}")
    print("=" * 70)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

    # Build model
    print("=> Creating model '{}'".format(args.arch))
    model = SimCLR(
        base_encoder=models.__dict__[args.arch],
        dim=args.simclr_dim,
    )
    model.cuda()

    # Loss and optimizer
    criterion = NTXentLoss(temperature=args.temperature).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cuda")
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    # Data loading
    traindir = os.path.join(args.data, "train")
    augmentation = build_augmentation(args)

    train_dataset = datasets.ImageFolder(
        traindir, SimCLRTransform(augmentation)
    )

    print(f"=> Training dataset: {len(train_dataset)} images")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        loss_avg = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]  "
            f"Loss: {loss_avg:.4f}  "
            f"LR: {lr:.6f}"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                args.save_dir,
                f"checkpoint_{epoch + 1:04d}.pth.tar",
            )

    print("\n=> Training complete!")
    print(f"   Final checkpoint saved in: {args.save_dir}")


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch."""
    batch_time = AverageMeter("Time", ":.3f")
    data_time = AverageMeter("Data", ":.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    model.train()
    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # Both views through the same encoder
        z1 = model(images[0])  # (N, dim)
        z2 = model(images[1])  # (N, dim)
        z = torch.cat([z1, z2], dim=0)  # (2N, dim)

        loss = criterion(z)

        losses.update(loss.item(), images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


if __name__ == "__main__":
    main()
