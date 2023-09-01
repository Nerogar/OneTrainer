from enum import Enum


class NoiseScheduler(Enum):
    DDIM = 'DDIM'
    EULER = 'EULER'
    EULER_A = 'EULER_A'

    def __str__(self):
        return self.value
