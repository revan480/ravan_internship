from torchvision import transforms

from .base import TransformationSpec


class HorizontalFlipSpec(TransformationSpec):
    name = "hflip"
    num_classes = 2

    def make_transform(self, label):
        if label == 1:
            return transforms.RandomHorizontalFlip(p=1.0)
        return transforms.Lambda(lambda x: x)

    def get_excluded_base_transforms(self):
        return {"hflip"}
