import argparse
import math
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models

from simclr.builder import SimCLR, SimCLRPred
from simclr.loader import BaselineTransform, ESSLTransform
from simclr.loss import NTXentLoss
from transformations import get_transformation_spec


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-SSL Figure 1: SimCLR Pre-Training with Invariance/Sensitivity Conditions"
    )

    # E-SSL experiment
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="vanilla SimCLR baseline (no tested transformation, no pred head)",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default=None,
        choices=["hflip", "grayscale", "rotation", "vflip", "jigsaw", "blur", "invert"],
        help="which transformation to test",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["invariance", "sensitivity"],
        help="invariance (always-apply, no pred) or sensitivity (stochastic + pred head)",
    )
    parser.add_argument(
        "--pred-lambda",
        type=float,
        default=0.5,
        help="weight for prediction loss (sensitivity mode only)",
    )

    # Data
    parser.add_argument("--data", type=str, default="./imagenet100")

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.3, help="base LR (scaled by batch_size/256)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=16)

    # SimCLR
    parser.add_argument("--temperature", type=float, default=0.5)

    # LR schedule
    parser.add_argument("--cos", action="store_true", default=True)
    parser.add_argument("--no-cos", action="store_false", dest="cos")
    parser.add_argument("--schedule", nargs="+", type=int, default=[300, 400])

    # Augmentation
    parser.add_argument("--color-strength", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=50)

    # Misc
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Scale learning rate
    args.lr = args.lr * args.batch_size / 256

    # Validate: either --baseline or both --transformation and --condition
    if not args.baseline:
        if args.transformation is None or args.condition is None:
            parser.error("--transformation and --condition are required unless --baseline is set")

    # Auto-generate save dir if not specified
    if not args.save_dir:
        if args.baseline:
            args.save_dir = "./checkpoints/baseline"
        else:
            args.save_dir = f"./checkpoints/{args.transformation}_{args.condition}"

    return args


class AverageMeter:
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

    # Resolve spec (None for baseline)
    if args.baseline:
        spec = None
    else:
        spec = get_transformation_spec(args.transformation)

    # Print configuration
    print("=" * 70)
    if args.baseline:
        print("E-SSL Figure 1: baseline (vanilla SimCLR)")
    else:
        print(f"E-SSL Figure 1: {args.transformation} / {args.condition}")
    print("=" * 70)
    print(f"  Architecture:     {args.arch}")
    if args.baseline:
        print(f"  Mode:             baseline (no tested transformation, no pred head)")
    else:
        print(f"  Transformation:   {args.transformation} ({spec.num_classes} classes)")
        print(f"  Condition:        {args.condition}")
        if args.condition == "sensitivity":
            print(f"  Pred lambda:      {args.pred_lambda}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr:.4f} (base 0.3 * {args.batch_size}/256)")
    print(f"  LR schedule:      {'cosine' if args.cos else f'step decay at {args.schedule}'}")
    print(f"  Temperature:      {args.temperature}")
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
    print(f"=> Creating model '{args.arch}'")
    if args.baseline or args.condition == "invariance":
        model = SimCLR(
            base_encoder=models.__dict__[args.arch],
            dim=128,
        )
    else:
        model = SimCLRPred(
            base_encoder=models.__dict__[args.arch],
            dim=128,
            num_pred_classes=spec.num_classes,
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

    # Resume
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
    if args.baseline:
        train_transform = BaselineTransform(color_strength=args.color_strength)
    else:
        train_transform = ESSLTransform(
            spec, args.condition, color_strength=args.color_strength
        )

    train_dataset = datasets.ImageFolder(traindir, train_transform)
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
    use_invariance_loop = args.baseline or args.condition == "invariance"

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        if use_invariance_loop:
            loss_avg = train_one_epoch_invariance(
                train_loader, model, criterion, optimizer, epoch, args
            )
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss: {loss_avg:.4f}  "
                f"LR: {lr:.6f}"
            )
        else:
            loss_avg, pred_loss_avg, pred_acc_avg = train_one_epoch_sensitivity(
                train_loader, model, criterion, optimizer, epoch, args
            )
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss: {loss_avg:.4f}  "
                f"Pred_Loss: {pred_loss_avg:.4f}  "
                f"Pred_Acc: {pred_acc_avg:.2f}%  "
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


def train_one_epoch_invariance(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch — invariance condition (no prediction head)."""
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

    for i, (data, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # data = [view1_batch, view2_batch]
        view1 = data[0].cuda(non_blocking=True)
        view2 = data[1].cuda(non_blocking=True)

        z1 = model(view1)
        z2 = model(view2)
        z = torch.cat([z1, z2], dim=0)

        loss = criterion(z)
        losses.update(loss.item(), view1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def train_one_epoch_sensitivity(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch — sensitivity condition (with prediction head)."""
    batch_time = AverageMeter("Time", ":.3f")
    data_time = AverageMeter("Data", ":.3f")
    losses = AverageMeter("Loss", ":.4f")
    pred_losses = AverageMeter("Pred_Loss", ":.4f")
    aug_accs = AverageMeter("Pred_Acc", ":.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, pred_losses, aug_accs],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    model.train()
    end = time.time()

    for i, (data, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # data = [view1_batch, view2_batch, aug_labels_batch]
        view1 = data[0].cuda(non_blocking=True)
        view2 = data[1].cuda(non_blocking=True)
        aug_labels = data[2].cuda(non_blocking=True)

        z1, aug_logits1 = model(view1)
        z2, aug_logits2 = model(view2)

        # Contrastive loss
        z = torch.cat([z1, z2], dim=0)
        contrastive_loss = criterion(z)

        # Prediction loss (both views)
        aug_logits = torch.cat([aug_logits1, aug_logits2], dim=0)
        aug_targets = torch.cat([aug_labels, aug_labels], dim=0)
        pred_loss = F.cross_entropy(aug_logits, aug_targets)

        # Total loss
        loss = contrastive_loss + args.pred_lambda * pred_loss

        # Prediction accuracy
        pred_acc = (aug_logits.argmax(1) == aug_targets).float().mean().item() * 100

        batch_size = view1.size(0)
        losses.update(loss.item(), batch_size)
        pred_losses.update(pred_loss.item(), batch_size)
        aug_accs.update(pred_acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, pred_losses.avg, aug_accs.avg


if __name__ == "__main__":
    main()
