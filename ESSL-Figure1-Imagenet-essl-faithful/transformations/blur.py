from PIL import ImageFilter

from .base import TransformationSpec


# 4 discrete blur levels. Kernel 0 = no blur (identity).
KERNEL_SIZES = [0, 5, 9, 15]


class FixedGaussianBlur:
    """Gaussian blur with a fixed kernel size (via PIL radius = kernel_size / 2)."""

    def __init__(self, kernel_size):
        self.radius = kernel_size / 2

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class BlurSpec(TransformationSpec):
    name = "blur"
    num_classes = 4

    def make_transform(self, label):
        kernel_size = KERNEL_SIZES[label]
        if kernel_size == 0:
            return lambda x: x
        return FixedGaussianBlur(kernel_size)

    def get_excluded_base_transforms(self):
        return {"blur"}
