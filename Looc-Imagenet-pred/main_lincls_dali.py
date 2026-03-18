"""
Linear evaluation with NVIDIA DALI — GPU-accelerated data loading.
JPEG decoding, resizing, cropping ALL happen on GPU instead of CPU.

Usage:
    python main_lincls_dali.py \
        --data ./imagenet100 \
        --pretrained ./checkpoints/exp_looc_cr_pred_05/checkpoint_0500.pth.tar \
        --looc-backbone --eval-rotation \
        --batch-size 512 --epochs 200
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.models as models

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="DALI-accelerated Linear Evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=30.0)
    parser.add_argument("--schedule", nargs="+", type=int, default=[120, 160])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--looc-backbone", action="store_true")
    parser.add_argument("--eval-rotation", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def get_file_list(data_dir):
    """Build list of (filepath, class_label). No rotation duplication."""
    files = []
    labels = []
    class_dirs = sorted([d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}

    for class_name, class_idx in class_to_idx.items():
        class_path = os.path.join(data_dir, class_name)
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                files.append(os.path.join(class_path, fname))
                labels.append(class_idx)

    return files, labels, len(class_to_idx)


@pipeline_def
def train_pipeline(file_list, label_list):
    """Training: decode + random_resized_crop + flip + normalize on GPU."""
    jpegs, labels = fn.readers.file(
        files=file_list, labels=label_list,
        random_shuffle=True, name="Reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(images, size=[224, 224],
                                     random_area=[0.2, 1.0], device="gpu")
    coin = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=coin, device="gpu")
    images = fn.crop_mirror_normalize(
        images, dtype=types.FLOAT, output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


@pipeline_def
def val_pipeline(file_list, label_list):
    """Val: decode + resize(256) + center_crop(224) + normalize on GPU."""
    jpegs, labels = fn.readers.file(
        files=file_list, labels=label_list,
        random_shuffle=False, name="Reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_shorter=256, device="gpu")
    images = fn.crop(images, crop_h=224, crop_w=224,
                     crop_pos_x=0.5, crop_pos_y=0.5, device="gpu")
    images = fn.crop_mirror_normalize(
        images, dtype=types.FLOAT, output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


def apply_rotation_batch(images, rot_class):
    """Apply rotation on GPU. rot_class: 0=0deg, 1=90deg, 2=180deg, 3=270deg."""
    if rot_class == 0:
        return images
    return torch.rot90(images, k=rot_class, dims=[2, 3])


class AverageMeter:
    def __init__(self):
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


def load_weights(model, checkpoint_path, looc_backbone=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    prefix = "backbone_q." if looc_backbone else "encoder_q."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            if new_key.startswith("fc.") or new_key.startswith("aug_"):
                continue
            new_state_dict[new_key] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded {len(new_state_dict)} keys from {prefix}*")
    print(f"  Missing keys: {msg.missing_keys}")
    return model


def train_epoch_object(model, train_loader, criterion, optimizer, epoch, print_freq):
    model.fc.train()
    losses, top1 = AverageMeter(), AverageMeter()

    for i, data in enumerate(train_loader):
        images = data[0]["data"]
        target = data[0]["label"].squeeze().long()

        output = model(images)
        loss = criterion(output, target)

        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"  Train: [{epoch}][{i}/{len(train_loader)}]  "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})")

    train_loader.reset()
    return losses.avg, top1.avg


def train_epoch_rotation(model, train_loader, criterion, optimizer, epoch, print_freq):
    """Each batch: DALI loads images, then we rotate 4 ways on GPU with torch.rot90."""
    model.fc.train()
    losses, top1 = AverageMeter(), AverageMeter()
    step = 0

    for i, data in enumerate(train_loader):
        images_orig = data[0]["data"]  # [N, 3, 224, 224] already on GPU
        batch_size = images_orig.size(0)

        for rot_class in range(4):
            images_rot = apply_rotation_batch(images_orig, rot_class)
            target = torch.full((batch_size,), rot_class, dtype=torch.long,
                                device=images_rot.device)

            output = model(images_rot)
            loss = criterion(output, target)

            acc1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % print_freq == 0:
            print(f"  Train: [{epoch}][{i}/{len(train_loader)}]  "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})")

    train_loader.reset()
    return losses.avg, top1.avg


def validate_object(model, val_loader, criterion):
    model.eval()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for data in val_loader:
            images = data[0]["data"]
            target = data[0]["label"].squeeze().long()

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

    val_loader.reset()
    return losses.avg, top1.avg, top5.avg


def validate_rotation(model, val_loader, criterion):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for data in val_loader:
            images_orig = data[0]["data"]
            batch_size = images_orig.size(0)

            for rot_class in range(4):
                images_rot = apply_rotation_batch(images_orig, rot_class)
                target = torch.full((batch_size,), rot_class, dtype=torch.long,
                                    device=images_rot.device)

                output = model(images_rot)
                loss = criterion(output, target)

                acc1, = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), batch_size)
                top1.update(acc1[0].item(), batch_size)

    val_loader.reset()
    return losses.avg, top1.avg, 100.0


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    print("=" * 60)
    print("  DALI-Accelerated Linear Evaluation")
    print("=" * 60)
    print(f"  Data:          {args.data}")
    print(f"  Checkpoint:    {args.pretrained}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  GPU:           {args.gpu}")
    print(f"  DALI threads:  {args.num_threads}")
    print(f"  Rotation eval: {args.eval_rotation}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  LR:            {args.lr} → schedule {args.schedule}")
    print()

    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    print("Building file lists...")
    train_files, train_labels, num_obj_classes = get_file_list(traindir)
    val_files, val_labels, _ = get_file_list(valdir)

    num_classes = 4 if args.eval_rotation else num_obj_classes
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    if args.eval_rotation:
        print(f"  Mode: Rotation (4 classes) — each image ×4 rotations on GPU")
    else:
        print(f"  Mode: Object ({num_classes} classes)")
    print()

    print("Building DALI pipelines...")
    train_pipe = train_pipeline(
        file_list=train_files, label_list=train_labels,
        batch_size=args.batch_size, num_threads=args.num_threads,
        device_id=args.gpu, seed=42,
    )
    train_pipe.build()

    val_pipe = val_pipeline(
        file_list=val_files, label_list=val_labels,
        batch_size=args.batch_size, num_threads=args.num_threads,
        device_id=args.gpu, seed=42,
    )
    val_pipe.build()

    train_loader = DALIClassificationIterator(
        train_pipe, reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP, auto_reset=False,
    )
    val_loader = DALIClassificationIterator(
        val_pipe, reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=False,
    )
    print(f"  DALI ready!")
    print()

    print("Loading pre-trained backbone...")
    model = models.__dict__[args.arch]()
    model = load_weights(model, args.pretrained, args.looc_backbone)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda(args.gpu)
    model.eval()
    model.fc.train()

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    print(f"  fc: Linear({model.fc.in_features}, {num_classes})")
    print()

    train_fn = train_epoch_rotation if args.eval_rotation else train_epoch_object
    val_fn = validate_rotation if args.eval_rotation else validate_object

    best_acc1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = args.lr
        for milestone in args.schedule:
            if epoch > milestone:
                lr *= 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, epoch, args.print_freq)
        val_loss, val_acc1, val_acc5 = val_fn(model, val_loader, criterion)

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc@1: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc@1: {val_acc1:.2f}%  "
              f"Val Acc@5: {val_acc5:.2f}%  "
              f"Time: {elapsed/60:.1f}min"
              f"{'  *BEST*' if is_best else ''}")

    task = "Rotation Classification (4 classes)" if args.eval_rotation else f"Object Classification ({num_classes} classes)"
    total_time = (time.time() - start_time) / 60
    print()
    print("=" * 70)
    print(f"FINAL RESULTS — {task}")
    print(f"  Best Val Acc@1: {best_acc1:.2f}%")
    print(f"  Checkpoint: {args.pretrained}")
    print(f"  Total time: {total_time:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()