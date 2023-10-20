from enum import Enum


class LearningRateScheduler(Enum):
    CONSTANT = 'CONSTANT'
    LINEAR = 'LINEAR'
    COSINE = 'COSINE'
    COSINE_WITH_RESTARTS = 'COSINE_WITH_RESTARTS'
    COSINE_WITH_HARD_RESTARTS = 'COSINE_WITH_HARD_RESTARTS'
    REX = 'REX'
    ADAFACTOR = 'ADAFACTOR'

    def __str__(self):
        return self.value
