from enum import Enum


class ImagePreprocessing(Enum):
    SQUARE_CENTER_CROP = 'SQUARE_CENTER_CROP'
    ASPECT_RATIO_BUCKETING = 'ASPECT_RATIO_BUCKETING'
    KEEP_ASPECT_RATIO = "KEEP_ASPECT_RATIO"

    def __str__(self):
        return self.value
