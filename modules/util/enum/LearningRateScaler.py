from enum import Enum

class LearningRateScaler(Enum):
    NONE = 'NONE'
    BATCH = 'BATCH'
    GRADIENT_ACCUMULATION = 'GRADIENT_ACCUMULATION'
    BOTH = 'BOTH'

    def __str__(self):
        return self.value