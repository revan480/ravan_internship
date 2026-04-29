"""
Base class for E-SSL transformation specifications.

Each transformation defines:
  - num_classes: how many discrete states the transformation has
  - sample_label(): uniform random label (for sensitivity mode)
  - sample_invariance_label(): label for invariance mode (always-apply for binary, uniform for multi-class)
  - make_transform(label): deterministic PIL transform for a given label
  - get_excluded_base_transforms(): base pipeline transforms to remove
"""

import random


class TransformationSpec:
    name = None
    num_classes = None

    def sample_label(self):
        """Uniform random over range(num_classes). Used in sensitivity mode."""
        return random.randint(0, self.num_classes - 1)

    def sample_invariance_label(self):
        """For binary transforms: always 1 (always apply).
        For multi-class: uniform random. Override in subclass if needed."""
        if self.num_classes == 2:
            return 1  # always apply
        return random.randint(0, self.num_classes - 1)

    def make_transform(self, label):
        """Return a deterministic PIL transform for the given label.
        label=0 is identity (no-op) for all transforms."""
        raise NotImplementedError

    def get_excluded_base_transforms(self):
        """Return set of base transform keys to remove.
        Valid keys: 'hflip', 'grayscale', 'blur'."""
        return set()
