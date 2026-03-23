import argparse
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
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Linear Evaluation for MoCo v2")

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

    # Pre-trained checkpoint
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="path to MoCo pre-trained checkpoint",
    )

    # Evaluation mode
    parser.add_argument(
        "--eval-rotation",
        action="store_true",
        default=False,
        help="evaluate rotation classification instead of object classification",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=30.0, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--schedule",
        nargs="+",
        type=int,
        default=[120, 160],
        help="LR drop epochs",
    )
    parser.add_argument("--workers", type=int, default=4, help="data loading workers")

    # Misc
    parser.add_argument("--print-freq", type=int, default=20, help="print frequency")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--resume", default="", type=str,
                        help="path to linear eval checkpoint to resume training")

    return parser.parse_args()


class RotationDataset(torch.utils.data.Dataset):
    """
    Wraps an image dataset to create a rotation classification task.
    Each image produces 4 copies rotated at 0, 90, 180, 270 degrees.
    Labels are the rotation class (0, 1, 2, 3).
    """

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

        # Get the original image (PIL)
        img_path, _ = self.base_dataset.samples[img_index]
        img = Image.open(img_path).convert("RGB")

        # Apply rotation
        if angle != 0:
            img = img.rotate(angle)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img, rotation_label


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top-k predictions."""
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


def save_checkpoint(state, filename="lincls_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_pretrained_weights(model, pretrained_path, args):
    """Load MoCo pre-trained weights into the model backbone."""
    print(f"=> Loading pre-trained checkpoint: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]

    # Extract encoder_q weights, strip prefix, skip fc (projection head).
    # Non-encoder_q keys are silently skipped — this includes encoder_k.*,
    # queue, queue_ptr, and aug_classifier.* (from Moco-Imagenet-pred checkpoints).
    new_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if not k.startswith("encoder_q."):
            continue
        # Strip the "encoder_q." prefix
        new_key = k[len("encoder_q."):]
        # Skip the MLP projection head (fc layers)
        if new_key.startswith("fc."):
            skipped.append(new_key)
            continue
        new_state_dict[new_key] = v

    # Load with strict=False (fc.weight and fc.bias will be missing)
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded {len(new_state_dict)} weight tensors")
    print(f"  Skipped projection head keys: {skipped}")
    print(f"  Missing keys (expected): {msg.missing_keys}")
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, (
        f"Unexpected missing keys: {msg.missing_keys}"
    )
    print("=> Pre-trained weights loaded successfully")


def main():
    args = parse_args()

    # Auto-detect number of object classes from dataset
    traindir = os.path.join(args.data, "train")
    object_classes = len([d for d in os.listdir(traindir) if os.path.isdir(os.path.join(traindir, d))])
    num_classes = 4 if args.eval_rotation else object_classes
    task_name = f"Rotation Classification (4 classes)" if args.eval_rotation else f"Object Classification ({object_classes} classes)"

    # Print configuration
    print("=" * 70)
    print(f"Linear Evaluation: {task_name}")
    print("=" * 70)
    print(f"  Architecture:   {args.arch}")
    print(f"  Pre-trained:    {args.pretrained}")
    print(f"  Num classes:    {num_classes}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LR schedule:    {args.schedule}")
    print(f"  Data:           {args.data}")
    print(f"  Seed:           {args.seed}")
    print("=" * 70)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

    # Build model
    print(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch]()

    # Load pre-trained weights (backbone only)
    load_pretrained_weights(model, args.pretrained, args)

    # Re-initialize the fc layer for the target task
    feature_dim = model.fc.in_features
    model.fc = nn.Linear(feature_dim, num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # Freeze all backbone parameters, only train fc
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.cuda()

    # Loss and optimizer (only optimize fc parameters)
    criterion = nn.CrossEntropyLoss().cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight and fc.bias
    optimizer = torch.optim.SGD(
        parameters,
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Data transforms
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

    # Build datasets
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    if args.eval_rotation:
        # Rotation classification datasets
        base_train = datasets.ImageFolder(traindir)
        base_val = datasets.ImageFolder(valdir)
        train_dataset = RotationDataset(base_train, transform=train_transform)
        val_dataset = RotationDataset(base_val, transform=test_transform)
    else:
        # Object classification datasets
        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
        val_dataset = datasets.ImageFolder(valdir, transform=test_transform)

    print(f"=> Train dataset: {len(train_dataset)} samples")
    print(f"=> Val dataset:   {len(val_dataset)} samples")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Resume from linear eval checkpoint
    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Resuming linear eval from '{args.resume}'")
            ckpt = torch.load(args.resume, map_location="cpu")
            start_epoch = ckpt["epoch"]
            best_acc1 = ckpt.get("best_acc1", 0.0)
            model.fc.load_state_dict(ckpt["fc_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"=> Resumed from epoch {start_epoch}, best_acc1={best_acc1:.2f}%")
        else:
            print(f"=> WARNING: no checkpoint found at '{args.resume}', starting from scratch")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc1 = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args
        )

        val_loss, val_acc1, val_acc5 = evaluate(val_loader, model, criterion, num_classes)

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc@1: {train_acc1:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc@1: {val_acc1:.2f}%  Val Acc@5: {val_acc5:.2f}%"
            + ("  *BEST*" if is_best else "")
        )

        # Save linear eval checkpoint for resume
        lincls_ckpt_path = os.path.join(os.path.dirname(args.pretrained) if args.pretrained else ".", "lincls_checkpoint.pth.tar")
        save_checkpoint({
            "epoch": epoch + 1,
            "fc_state_dict": model.fc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": best_acc1,
            "args": vars(args),
        }, filename=lincls_ckpt_path)

        if is_best:
            best_ckpt_path = lincls_ckpt_path.replace("lincls_checkpoint", "lincls_best")
            save_checkpoint({
                "epoch": epoch + 1,
                "fc_state_dict": model.fc.state_dict(),
                "best_acc1": best_acc1,
            }, filename=best_ckpt_path)

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS — {task_name}")
    print(f"  Best Val Acc@1: {best_acc1:.2f}%")
    print(f"  Checkpoint: {args.pretrained}")
    print("=" * 70)


def adjust_learning_rate(optimizer, epoch, args):
    """Step decay LR schedule."""
    lr = args.lr
    for milestone in args.schedule:
        if epoch >= milestone:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch."""
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":.2f")

    model.train()
    # Freeze BN layers (backbone is frozen, so BN should be in eval mode)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(
                f"  Train: [{epoch + 1}][{i}/{len(train_loader)}]  "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})"
            )

    return losses.avg, top1.avg


def evaluate(val_loader, model, criterion, num_classes):
    """Evaluate on the validation set."""
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":.2f")
    top5 = AverageMeter("Acc@5", ":.2f")

    topk = (1, min(5, num_classes))

    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=topk)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
