"""
Data loading utilities for MoCo v2.

Provides:
  - TwoCropsTransform: applies a base transform twice to produce two views
  - GaussianBlur: random Gaussian blur augmentation
"""

import random

from PIL import ImageFilter


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
