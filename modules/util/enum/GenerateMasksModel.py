from enum import Enum


class GenerateMasksModel(Enum):
    CLIPSEG = 'CLIPSEG'
    REMBG = 'REMBG'

    def __str__(self):
        return self.value
