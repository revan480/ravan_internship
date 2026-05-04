from torchvision import transforms

from .base import TransformationSpec


class VerticalFlipSpec(TransformationSpec):
    name = "vflip"
    num_classes = 2

    def make_transform(self, label):
        if label == 1:
            return transforms.RandomVerticalFlip(p=1.0)
        return transforms.Lambda(lambda x: x)
