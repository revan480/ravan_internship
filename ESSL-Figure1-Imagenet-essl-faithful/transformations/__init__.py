from .horizontal_flip import HorizontalFlipSpec
from .vertical_flip import VerticalFlipSpec
from .grayscale import GrayscaleSpec
from .rotation import RotationSpec
from .jigsaw import JigsawSpec
from .blur import BlurSpec
from .color_inversion import ColorInversionSpec

REGISTRY = {
    "hflip": HorizontalFlipSpec,
    "vflip": VerticalFlipSpec,
    "grayscale": GrayscaleSpec,
    "rotation": RotationSpec,
    "jigsaw": JigsawSpec,
    "blur": BlurSpec,
    "invert": ColorInversionSpec,
}


def get_transformation_spec(name):
    """Look up a TransformationSpec by name."""
    return REGISTRY[name]()
