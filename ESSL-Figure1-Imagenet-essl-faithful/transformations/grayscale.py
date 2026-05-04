from torchvision import transforms

from .base import TransformationSpec


class GrayscaleSpec(TransformationSpec):
    name = "grayscale"
    num_classes = 2

    def make_transform(self, label):
        if label == 1:
            return transforms.Grayscale(num_output_channels=3)
        return transforms.Lambda(lambda x: x)

    def get_excluded_base_transforms(self):
        return {"grayscale"}
