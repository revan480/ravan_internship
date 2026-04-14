"""
Data loading utilities for BYOL.

Provides:
  - BYOLTransform: applies a base transform twice to produce two views
  - GaussianBlur: random Gaussian blur augmentation
"""

import random

from PIL import ImageFilter


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
