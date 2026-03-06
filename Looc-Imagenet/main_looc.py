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

from looc.builder import LooC
from looc.loader import LooCTransform


def parse_args():
    parser = argparse.ArgumentParser(description="LooC Pre-Training (Single GPU)")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="./imagenet100",
        help="path to ImageNet-100 dataset",
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
    parser.add_argument("--lr", type=float, default=0.03, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--workers", type=int, default=4, help="data loading workers")

    # MoCo / LooC
    parser.add_argument("--moco-dim", type=int, default=128, help="feature dimension")
    parser.add_argument("--moco-k", type=int, default=16384, help="queue size per head")
    parser.add_argument("--moco-m", type=float, default=0.999, help="momentum for key encoder")
    parser.add_argument("--moco-t", type=float, default=0.2, help="temperature")
    parser.add_argument("--n-aug", type=int, default=2, help="number of atomic augmentations (default: 2 for rotation + color)")

    # LR schedule
    parser.add_argument("--cos", action="store_true", default=False, help="use cosine LR schedule")
    parser.add_argument("--schedule", nargs="+", type=int, default=[300, 400], help="LR drop epochs (step decay)")

    # Augmentation
    parser.add_argument("--color-strength", type=float, default=1.0, help="multiplier for color jittering strength")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./checkpoints/default", help="directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")
    parser.add_argument("--save-freq", type=int, default=50, help="save checkpoint every N epochs")

    # Misc
    parser.add_argument("--print-freq", type=int, default=20, help="print frequency (batches)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()


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


def adjust_learning_rate(optimizer, epoch, args):
    """LR schedule: step decay or cosine annealing."""
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


def contrastive_accuracy(output, target):
    """Computes the accuracy of the contrastive prediction (is the positive the highest?)."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        return acc


def save_checkpoint(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"  => Saved checkpoint: {filepath}")


def main():
    args = parse_args()

    # Print experiment configuration
    print("=" * 70)
    print("LooC Pre-Training Configuration")
    print("=" * 70)
    print(f"  Architecture:     {args.arch}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  LR schedule:      {'cosine' if args.cos else f'step decay at {args.schedule}'}")
    print(f"  MoCo dim:         {args.moco_dim}")
    print(f"  MoCo K (queue):   {args.moco_k}")
    print(f"  MoCo m (momentum):{args.moco_m}")
    print(f"  MoCo T (temp):    {args.moco_t}")
    print(f"  n_aug:            {args.n_aug} (heads: {args.n_aug + 1})")
    print(f"  Color strength:   {args.color_strength}")
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
    model = LooC(
        base_encoder=models.__dict__[args.arch],
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        n_aug=args.n_aug,
    )
    model.cuda()

    # Optimizer
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

    train_dataset = datasets.ImageFolder(
        traindir, LooCTransform(color_strength=args.color_strength)
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

        loss_avg, acc_avg = train_one_epoch(
            train_loader, model, optimizer, epoch, args
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]  "
            f"Loss: {loss_avg:.4f}  "
            f"Acc: {acc_avg:.2f}%  "
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


def train_one_epoch(train_loader, model, optimizer, epoch, args):
    """Train for one epoch."""
    batch_time = AverageMeter("Time", ":.3f")
    data_time = AverageMeter("Data", ":.3f")
    losses = AverageMeter("Loss", ":.4f")
    accs = AverageMeter("Acc", ":.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accs],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    model.train()
    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Move all 4 views to GPU
        views = [v.cuda(non_blocking=True) for v in images]

        # Forward pass (loss computed inside model)
        loss, logits, labels = model(views)

        acc = contrastive_accuracy(logits, labels)

        losses.update(loss.item(), views[0].size(0))
        accs.update(acc.item(), views[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, accs.avg


if __name__ == "__main__":
    main()
