from enum import Enum


class BalancingStrategy(Enum):
    REPEATS = 'REPEATS'
    SAMPLES = 'SAMPLES'

    def __str__(self):
        return self.value
