from enum import Enum


class TensorboardMode(Enum):
    OFF = 'OFF'
    ALWAYS_ON = 'ALWAYS_ON'
    TRAIN_ONLY = 'TRAIN_ONLY'

    def __str__(self):
        return self.value
