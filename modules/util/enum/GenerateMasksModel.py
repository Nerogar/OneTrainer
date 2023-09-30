from enum import Enum


class GenerateMasksModel(Enum):
    CLIPSEG = 'CLIPSEG'
    REMBG = 'REMBG'
    COLOR = 'COLOR'

    def __str__(self):
        return self.value
