"""
Data loading utilities for MoCo v2 with augmentation prediction.

Provides:
  - TwoCropsTransform: applies a base transform twice to produce two views
  - GaussianBlur: random Gaussian blur augmentation
  - AugPredQueryTransform: query transform that tracks which augmentations were applied
  - MoCoAugPredDataset: dataset wrapper returning (query, key, aug_label)
"""

import random

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter


class TwoCropsTransform:
    """Take one image, apply the base transform twice independently, return [view1, view2]."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation as used in MoCo v2 (SimCLR style)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomRotation90:
    """Apply a random rotation from {90, 180, 270} degrees (never 0)."""

    def __call__(self, img):
        angle = random.choice([90, 180, 270])
        return img.rotate(angle)


class AugPredQueryTransform:
    """
    Query transform that applies augmentations stochastically and tracks
    which combination was applied.

    Augmentation pipeline (matches build_augmentation order in main_moco.py):
        1. Rotation (p=0.5): random angle from {90, 180, 270}
        2. RandomResizedCrop(224) + RandomHorizontalFlip (always)
        3. ColorJitter (p=0.8) + RandomGrayscale (p=0.2) — both part of "color" group
        4. GaussianBlur (p=0.5), ToTensor, Normalize (always)

    Returns:
        (tensor, aug_label) where aug_label encodes the combination:
            0 = neither rotation nor color applied
            1 = rotation only
            2 = color only
            3 = both rotation and color
    """

    def __init__(self, use_rotation=True, use_color=True, color_strength=1.0):
        self.use_rotation = use_rotation
        self.use_color = use_color

        # Rotation
        self.rotation_transform = RandomRotation90()
        self.p_rotation = 0.5

        # Color
        s = color_strength
        self.color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        self.p_color = 0.8
        self.p_grayscale = 0.2

        # Always-applied transforms
        self.crop_flip = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
        self.blur = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
        self.final = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        applied_rot = False
        applied_color = False

        # 1. Rotation (before crop, matching build_augmentation order)
        if self.use_rotation and random.random() < self.p_rotation:
            img = self.rotation_transform(img)
            applied_rot = True

        # 2. Crop + flip (always)
        img = self.crop_flip(img)

        # 3. Color jitter + grayscale (both part of "color augmentation" group)
        if self.use_color:
            if random.random() < self.p_color:
                img = self.color_jitter(img)
                applied_color = True
            if random.random() < self.p_grayscale:
                img = transforms.functional.to_grayscale(img, num_output_channels=3)
                applied_color = True  # grayscale counts as color aug

        # 4. Blur + tensor + normalize (always)
        img = self.blur(img)
        img = self.final(img)

        # Encode label: 0=neither, 1=rot_only, 2=color_only, 3=both
        aug_label = int(applied_rot) + 2 * int(applied_color)
        return img, aug_label


class MoCoAugPredDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies separate query/key transforms and returns
    (query_image, key_image, aug_label).

    The query transform tracks which augmentations were applied (returns tensor + label).
    The key transform is a standard augmentation pipeline (returns tensor only).
    """

    def __init__(self, root, query_transform, key_transform):
        self.dataset = datasets.ImageFolder(root)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, _ = self.dataset.samples[index]
        img = Image.open(path).convert("RGB")

        query_img, aug_label = self.query_transform(img)
        key_img = self.key_transform(img)
        return query_img, key_img, aug_label
