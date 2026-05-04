from .base import TransformationSpec


ANGLES = [0, 90, 180, 270]


class RotationSpec(TransformationSpec):
    name = "rotation"
    num_classes = 4

    def make_transform(self, label):
        angle = ANGLES[label]
        if angle == 0:
            return lambda x: x
        return lambda x: x.rotate(angle)
