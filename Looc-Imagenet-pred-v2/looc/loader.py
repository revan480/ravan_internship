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

Prediction modes (pred_mode):
  "angle" (default): rotation always applied, label = angle class (0=90, 1=180, 2=270)
  "combo": rotation/color stochastic, label = int(rot) + 2*int(color)
           (0=neither, 1=rot only, 2=color only, 3=both)
"""

import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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

    pred_mode controls augmentation application and label:
        "angle": rotation always applied, label = rotation angle class (0/1/2)
        "combo": rotation p=0.5, color p=0.8, grayscale p=0.2,
                 label = int(rot_applied) + 2*int(color_applied)
    """

    def __init__(self, aug_types=None, color_strength=1.0, pred_mode="angle"):
        if aug_types is None:
            aug_types = ["rotation", "color"]
        self.aug_types = aug_types
        self.pred_mode = pred_mode
        self.crop = transforms.RandomResizedCrop(224, scale=(0.2, 1.0))
        self.flip = transforms.RandomHorizontalFlip()
        self.blur = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        s = color_strength
        # RandomApply-wrapped version (used in angle mode)
        self.color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8
        )
        self.grayscale = transforms.RandomGrayscale(p=0.2)
        # Raw ColorJitter without RandomApply (used in combo mode for explicit control)
        self.raw_color_jitter = transforms.ColorJitter(
            0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s
        )

    def _apply_base(self, img):
        """Apply base augmentations (always independent): crop, flip."""
        img = self.crop(img)
        img = self.flip(img)
        return img

    def _apply_rotation(self, img, angle):
        """Apply rotation. angle=0 means no rotation."""
        if angle == 0:
            return img
        return img.rotate(angle)

    def _apply_color(self, img, seed):
        """Apply color jitter + grayscale using saved RNG seed for reproducibility.

        Used in angle mode. Saves and restores RNG state so color augmentation
        can be replayed deterministically when two views need to share the same
        color transform.
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

    def _apply_color_combo(self, img, seed, apply_jitter, apply_grayscale):
        """Apply color with explicit control over what fires (combo mode).

        Instead of relying on RandomApply's internal probability, the caller
        pre-decides whether jitter/grayscale should fire. The seed ensures
        deterministic ColorJitter parameters when sharing between views.
        """
        torch_state = torch.random.get_rng_state()
        py_state = random.getstate()
        torch.random.manual_seed(seed)
        random.seed(seed)
        if apply_jitter:
            img = self.raw_color_jitter(img)
        if apply_grayscale:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)
        return img

    def _finalize(self, img):
        """ToTensor + Normalize."""
        return self.normalize(self.to_tensor(img))

    def _make_view(self, img, angle, color_seed):
        """Full pipeline for angle mode: rotation → crop → flip → color → blur → normalize."""
        v = self._apply_rotation(img.copy(), angle)
        v = self._apply_base(v)
        v = self._apply_color(v, color_seed)
        v = self.blur(v)
        return self._finalize(v)

    def _make_view_combo(self, img, angle, color_seed, apply_jitter, apply_grayscale):
        """Full pipeline for combo mode: rotation → crop → flip → color_combo → blur → normalize."""
        v = self._apply_rotation(img.copy(), angle)
        v = self._apply_base(v)
        v = self._apply_color_combo(v, color_seed, apply_jitter, apply_grayscale)
        v = self.blur(v)
        return self._finalize(v)

    def _sample_rot_combo(self):
        """Sample rotation decision for combo mode: p=0.5, returns (angle, applied)."""
        applied = random.random() < 0.5
        angle = random.choice([90, 180, 270]) if applied else 0
        return angle, applied

    def _sample_color_combo(self):
        """Sample color decisions for combo mode. Returns (seed, apply_jitter, apply_grayscale, applied)."""
        seed = random.randint(0, 2**31)
        apply_jitter = random.random() < 0.8
        apply_grayscale = random.random() < 0.2
        applied = apply_jitter or apply_grayscale
        return seed, apply_jitter, apply_grayscale, applied

    def __call__(self, img):
        if self.pred_mode == "combo":
            return self._call_combo(img)
        else:
            return self._call_angle(img)

    def _call_angle(self, img):
        """Original angle prediction mode: rotation always applied, label = angle class."""
        # Sample q's augmentation parameters (always both rotation and color)
        q_rot = random.choice([90, 180, 270])
        q_color = random.randint(0, 2**31)
        rot_label = {90: 0, 180: 1, 270: 2}[q_rot]

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

        return views + [rot_label]

    def _call_combo(self, img):
        """Combo prediction mode: stochastic rotation/color, label = int(rot) + 2*int(color).

        View sharing logic:
          k1 (shares rotation with q): same apply_rot + q_rot, fresh color decisions
          k2 (shares color with q): fresh rotation decisions, same apply_jitter + apply_grayscale + q_seed
        """
        # --- Sample q's augmentation decisions ---
        q_rot, q_rot_applied = self._sample_rot_combo()
        q_color_seed, q_apply_jitter, q_apply_grayscale, q_color_applied = self._sample_color_combo()
        combo_label = int(q_rot_applied) + 2 * int(q_color_applied)

        # --- Sample k0's decisions (all fresh) ---
        k0_rot, _ = self._sample_rot_combo()
        k0_color_seed, k0_apply_jitter, k0_apply_grayscale, _ = self._sample_color_combo()

        views = [
            self._make_view_combo(img, q_rot, q_color_seed, q_apply_jitter, q_apply_grayscale),    # q
            self._make_view_combo(img, k0_rot, k0_color_seed, k0_apply_jitter, k0_apply_grayscale), # k0
        ]

        # For each atomic augmentation, create a key that shares it with q
        for aug in self.aug_types:
            if aug == "rotation":
                # k1: shares rotation decision + angle with q, fresh color
                ki_rot = q_rot  # shared (includes the apply/not-apply decision)
                ki_seed, ki_jitter, ki_gray, _ = self._sample_color_combo()  # fresh
                views.append(self._make_view_combo(img, ki_rot, ki_seed, ki_jitter, ki_gray))
            else:  # "color"
                # k2: shares color decisions + seed with q, fresh rotation
                ki_rot, _ = self._sample_rot_combo()  # fresh
                views.append(self._make_view_combo(
                    img, ki_rot, q_color_seed, q_apply_jitter, q_apply_grayscale  # shared
                ))

        return views + [combo_label]
