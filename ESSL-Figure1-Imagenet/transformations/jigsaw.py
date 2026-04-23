import itertools

from PIL import Image

from .base import TransformationSpec


# All 24 permutations of 4 quadrants. Index 0 = identity (0,1,2,3).
PERMS = list(itertools.permutations([0, 1, 2, 3]))


class JigsawTransform:
    """PIL-level 2x2 jigsaw: split 224x224 into four 112x112 quadrants, permute."""

    def __init__(self, perm_idx):
        self.perm = PERMS[perm_idx]

    def __call__(self, img):
        w, h = img.size  # expected 224x224 after RandomResizedCrop
        hw, hh = w // 2, h // 2

        # Quadrants: 0=TL, 1=TR, 2=BL, 3=BR
        patches = [
            img.crop((0, 0, hw, hh)),
            img.crop((hw, 0, w, hh)),
            img.crop((0, hh, hw, h)),
            img.crop((hw, hh, w, h)),
        ]

        # Reassemble with permuted patches
        new_img = Image.new(img.mode, (w, h))
        positions = [(0, 0), (hw, 0), (0, hh), (hw, hh)]
        for dst_idx, src_idx in enumerate(self.perm):
            new_img.paste(patches[src_idx], positions[dst_idx])

        return new_img


class JigsawSpec(TransformationSpec):
    name = "jigsaw"
    num_classes = 24

    def make_transform(self, label):
        return JigsawTransform(label)
