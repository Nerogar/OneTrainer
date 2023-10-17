from enum import Enum


class AlignPropLoss(Enum):
    HPS = 'HPS'
    AESTHETIC = 'AESTHETIC'

    def __str__(self):
        return self.value
