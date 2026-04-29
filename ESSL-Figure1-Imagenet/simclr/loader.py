"""
Data loading utilities for E-SSL Figure 1 experiments.

Provides:
  - ESSLTransform: unified transform for both invariance and sensitivity conditions
  - GaussianBlur: random Gaussian blur augmentation (from existing SimCLR code)
"""

import random

from PIL import ImageFilter
import torchvision.transforms as transforms


class GaussianBlur:
    """Gaussian blur augmentation (SimCLR / MoCo v2 style)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class BaselineTransform:
    """Vanilla SimCLR augmentation: apply standard pipeline twice, return [view1, view2]."""

    def __init__(self, color_strength=1.0):
        s = color_strength
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, img):
        return [self.base_transform(img), self.base_transform(img)]


class ESSLTransform:
    """Augmentation transform for E-SSL Figure 1 experiments.

    Builds the SimCLR augmentation pipeline with the tested transformation
    inserted at the correct position, excluding conflicting base transforms.

    Invariance mode: each view gets an independent random application.
        Returns [view1, view2].
    Sensitivity mode: both views share the same transformation label.
        Returns [view1, view2, label].
    """

    def __init__(self, spec, condition, color_strength=1.0):
        """
        Args:
            spec: TransformationSpec instance
            condition: "invariance" or "sensitivity"
            color_strength: multiplier for color jittering (default 1.0)
        """
        self.spec = spec
        self.condition = condition
        self.color_strength = color_strength
        self.excluded = spec.get_excluded_base_transforms()

    def _build_pipeline(self, transform_fn):
        """Build the full augmentation pipeline with the tested transform inserted.

        Pipeline order:
            rotation → crop → hflip/vflip → colorjitter → grayscale →
            blur → invert → jigsaw → ToTensor → Normalize
        """
        aug_list = []

        # Position 1: Rotation (before crop)
        if self.spec.name == "rotation":
            aug_list.append(transform_fn)

        # Position 2: Crop (always)
        aug_list.append(transforms.RandomResizedCrop(224, scale=(0.2, 1.0)))

        # Position 3: Flips
        if self.spec.name == "hflip":
            aug_list.append(transform_fn)
        elif self.spec.name == "vflip":
            # Keep standard HFlip alongside tested VFlip
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transform_fn)
        elif "hflip" not in self.excluded:
            aug_list.append(transforms.RandomHorizontalFlip())

        # Position 4: Color jitter (always applied)
        s = self.color_strength
        aug_list.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                p=0.8,
            )
        )

        # Position 5: Grayscale
        if self.spec.name == "grayscale":
            aug_list.append(transform_fn)
        elif "grayscale" not in self.excluded:
            aug_list.append(transforms.RandomGrayscale(p=0.2))

        # Position 6: Blur
        if self.spec.name == "blur":
            aug_list.append(transform_fn)
        elif "blur" not in self.excluded:
            aug_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        # Position 7: Color inversion (PIL-level, before ToTensor)
        if self.spec.name == "invert":
            aug_list.append(transform_fn)

        # Position 8: Jigsaw (PIL-level, after crop ensures 224x224)
        if self.spec.name == "jigsaw":
            aug_list.append(transform_fn)

        # Position 9: ToTensor + Normalize
        aug_list.append(transforms.ToTensor())
        aug_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )

        return transforms.Compose(aug_list)

    def __call__(self, img):
        if self.condition == "invariance":
            # Each view gets an independent random transformation instance
            label1 = self.spec.sample_invariance_label()
            label2 = self.spec.sample_invariance_label()
            pipeline1 = self._build_pipeline(self.spec.make_transform(label1))
            pipeline2 = self._build_pipeline(self.spec.make_transform(label2))
            return [pipeline1(img), pipeline2(img)]
        else:
            # Sensitivity: shared label across both views
            label = self.spec.sample_label()
            transform_fn = self.spec.make_transform(label)
            pipeline = self._build_pipeline(transform_fn)
            return [pipeline(img), pipeline(img), label]
