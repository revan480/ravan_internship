"""
Data loading utilities for BYOL + Augmentation Prediction.

Provides:
  - BYOLTransform: applies a base transform twice to produce two views
  - BYOLPredTransform: applies augmentations with tracking, returns [view1, view2, aug_label]
  - GaussianBlur: random Gaussian blur augmentation
  - RandomRotation90: random rotation from {0, 90, 180, 270}
"""

import random

from PIL import ImageFilter
import torchvision.transforms as transforms


class BYOLTransform:
    """Take one image, apply the base transform twice independently, return [view1, view2]."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return [v1, v2]


class GaussianBlur:
    """Gaussian blur augmentation (SimCLR / MoCo v2 / BYOL style)."""

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


class BYOLPredTransform:
    """Augmentation transform with tracking for augmentation prediction.

    Stochastically decides whether to apply rotation and color augmentations,
    then applies the SAME decisions to BOTH views (with independent random params).

    Returns [view1, view2, aug_label] where:
        aug_label = int(rotation_applied) + 2 * int(color_applied)
        0 = neither, 1 = rotation only, 2 = color only, 3 = both
    """

    def __init__(self, use_rotation=True, use_color=True, color_strength=1.0):
        self.use_rotation = use_rotation
        self.use_color = use_color
        self.color_strength = color_strength

    def _build_transform(self, apply_rotation, apply_color):
        """Build augmentation pipeline based on the stochastic decisions."""
        aug_list = []

        # Rotation BEFORE crop (if decided to apply)
        if apply_rotation:
            aug_list.append(RandomRotation90())

        # Always: crop + flip
        aug_list.extend([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

        # Color augmentations (if decided to apply)
        if apply_color:
            s = self.color_strength
            aug_list.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                    p=0.8,
                )
            )
            aug_list.append(transforms.RandomGrayscale(p=0.2))

        # Always: blur + tensor + normalize
        aug_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        aug_list.append(transforms.ToTensor())
        aug_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        return transforms.Compose(aug_list)

    def __call__(self, x):
        # Stochastic augmentation decisions (shared for both views)
        applied_rot = self.use_rotation and random.random() < 0.5
        applied_color = self.use_color and random.random() < 0.8
        aug_label = int(applied_rot) + 2 * int(applied_color)

        # Build transform with these decisions
        transform = self._build_transform(applied_rot, applied_color)

        # Apply twice independently (same decisions, different random params)
        view1 = transform(x)
        view2 = transform(x)

        return [view1, view2, aug_label]
