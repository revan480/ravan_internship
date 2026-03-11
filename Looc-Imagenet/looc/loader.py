"""
Data loading utilities for LooC.

Provides:
  - LooCTransform: generates n_aug+2 views with controlled augmentation sharing
  - GaussianBlur: random Gaussian blur augmentation (same as MoCo v2)

Views depend on aug_types:
  aug_types=["rotation", "color"] (n_aug=2, 4 views):
    q:  rot=R1, color=C1  — query (all fresh)
    k0: rot=R2, color=C2  — standard key (all fresh, positive in Z0)
    k1: rot=R1, color=C3  — shares rotation with q (positive in Z1)
    k2: rot=R3, color=C1  — shares color with q (positive in Z2)

  aug_types=["rotation"] (n_aug=1, 3 views):
    q:  rot=R1, color=C1
    k0: rot=R2, color=C2
    k1: rot=R1, color=C3  — shares rotation with q

  aug_types=["color"] (n_aug=1, 3 views):
    q:  rot=R1, color=C1
    k0: rot=R2, color=C2
    k1: rot=R3, color=C1  — shares color with q
"""

import random

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter


class GaussianBlur:
    """Gaussian blur augmentation as used in MoCo v2 (SimCLR style)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class LooCTransform:
    """
    Generate n_aug+2 views for LooC training with controlled augmentation sharing.

    For each atomic augmentation in aug_types, one extra key view is created that
    shares THAT augmentation with the query. Augmentations not in aug_types are
    still applied but independently (as base augmentations).

    Pipeline per view (MoCo v2 order):
        rotation → crop → flip → color_jitter → grayscale → blur → tensor → normalize

    Rotation is always applied (random choice from {90, 180, 270}).
    Color jitter is replayed via RNG seed save/restore for deterministic sharing.
    Base augmentations (crop, flip, blur) are always sampled independently.
    """

    def __init__(self, aug_types=None, color_strength=1.0):
        if aug_types is None:
            aug_types = ["rotation", "color"]
        self.aug_types = aug_types
        self.crop = transforms.RandomResizedCrop(224, scale=(0.2, 1.0))
        self.flip = transforms.RandomHorizontalFlip()
        self.blur = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        s = color_strength
        self.color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8
        )
        self.grayscale = transforms.RandomGrayscale(p=0.2)

    def _apply_base(self, img):
        """Apply base augmentations (always independent): crop, flip."""
        img = self.crop(img)
        img = self.flip(img)
        return img

    def _apply_rotation(self, img, angle):
        """Apply rotation. Angle is always one of {90, 180, 270}."""
        return img.rotate(angle)

    def _apply_color(self, img, seed):
        """Apply color jitter + grayscale using saved RNG seed for reproducibility.

        Saves and restores RNG state so color augmentation can be replayed
        deterministically when two views need to share the same color transform.
        """
        torch_state = torch.random.get_rng_state()
        py_state = random.getstate()
        torch.random.manual_seed(seed)
        random.seed(seed)
        img = self.color_jitter(img)
        img = self.grayscale(img)
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)
        return img

    def _finalize(self, img):
        """ToTensor + Normalize."""
        return self.normalize(self.to_tensor(img))

    def _make_view(self, img, angle, color_seed):
        """Full pipeline: rotation → crop → flip → color → blur → tensor → normalize."""
        v = self._apply_rotation(img.copy(), angle)
        v = self._apply_base(v)
        v = self._apply_color(v, color_seed)
        v = self.blur(v)
        return self._finalize(v)

    def __call__(self, img):
        # Sample q's augmentation parameters (always both rotation and color)
        q_rot = random.choice([90, 180, 270])
        q_color = random.randint(0, 2**31)

        # Sample k0's parameters (all fresh)
        k0_rot = random.choice([90, 180, 270])
        k0_color = random.randint(0, 2**31)

        views = [
            self._make_view(img, q_rot, q_color),    # q
            self._make_view(img, k0_rot, k0_color),   # k0 (all fresh, positive in Z0)
        ]

        # For each atomic augmentation, create a key that shares it with q
        for aug in self.aug_types:
            if aug == "rotation":
                ki_rot = q_rot                          # shared with q
                ki_color = random.randint(0, 2**31)     # fresh
            else:  # "color"
                ki_rot = random.choice([90, 180, 270])  # fresh
                ki_color = q_color                      # shared with q
            views.append(self._make_view(img, ki_rot, ki_color))

        return views
