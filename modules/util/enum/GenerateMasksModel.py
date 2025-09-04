from enum import Enum


class GenerateMasksModel(Enum):
    CLIPSEG = 'CLIPSEG'
    REMBG = 'REMBG'
    REMBG_HUMAN = 'REMBG_HUMAN'
    COLOR = 'COLOR'
    SAMOONDREAM = 'SAMOONDREAM'

    def __str__(self):
        return self.value
