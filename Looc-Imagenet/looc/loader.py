"""
Data loading utilities for LooC.

Provides:
  - LooCTransform: generates 4 views with controlled augmentation parameter sharing
  - GaussianBlur: random Gaussian blur augmentation (same as MoCo v2)

The 4 views are:
  q:  rot=R1, color=C1  — query (all fresh)
  k0: rot=R2, color=C2  — standard key (all fresh, positive in Z0)
  k1: rot=R1, color=C3  — shares rotation with q (positive in Z1)
  k2: rot=R3, color=C1  — shares color with q (positive in Z2)
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
    Generate 4 views for LooC training with controlled augmentation parameter sharing.

    Views:
        q:  rot=R1 (fresh), color=C1 (fresh)   — query
        k0: rot=R2 (fresh), color=C2 (fresh)    — standard key (positive in Z0)
        k1: rot=R1 (SAME), color=C3 (fresh)     — positive in Z1 (shares rotation)
        k2: rot=R3 (fresh), color=C1 (SAME)     — positive in Z2 (shares color)

    Pipeline per view (MoCo v2 order):
        rotation → crop → flip → color_jitter → grayscale → blur → tensor → normalize

    Rotation is always applied (random choice from {90, 180, 270}).
    Color jitter is replayed via RNG seed save/restore for deterministic sharing.
    Base augmentations (crop, flip, blur) are always sampled independently.
    """

    def __init__(self, color_strength=1.0):
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
        # Sample 3 independent rotation angles (always applied)
        R1 = random.choice([90, 180, 270])
        R2 = random.choice([90, 180, 270])
        R3 = random.choice([90, 180, 270])

        # Sample 3 independent color seeds
        C1 = random.randint(0, 2**31)
        C2 = random.randint(0, 2**31)
        C3 = random.randint(0, 2**31)

        q = self._make_view(img, R1, C1)   # query
        k0 = self._make_view(img, R2, C2)  # all fresh — positive in Z0
        k1 = self._make_view(img, R1, C3)  # shares rotation with q — positive in Z1
        k2 = self._make_view(img, R3, C1)  # shares color with q — positive in Z2

        return [q, k0, k1, k2]
