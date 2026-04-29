from PIL import ImageOps

from .base import TransformationSpec


class ColorInversionSpec(TransformationSpec):
    name = "invert"
    num_classes = 2

    def make_transform(self, label):
        if label == 1:
            return lambda img: ImageOps.invert(img)
        return lambda x: x
