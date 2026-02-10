from enum import Enum


class ImagePreprocessing(Enum):
    ORIGINAL_SIZE = 'ORIGINAL_SIZE'
    SQUARE_CENTER_CROP = 'SQUARE_CENTER_CROP'
    ASPECT_RATIO_BUCKETING = 'ASPECT_RATIO_BUCKETING'

    def __str__(self):
        return self.value
