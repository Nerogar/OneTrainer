from enum import Enum

class LossScaler(Enum):
    NONE = 'NONE'
    BATCH = 'BATCH'
    GRADIENT_ACCUMULATION = 'GRADIENT_ACCUMULATION'
    BOTH = 'BOTH'

    def __str__(self):
        return self.value