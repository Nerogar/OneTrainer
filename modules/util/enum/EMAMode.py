from enum import Enum


class EMAMode(Enum):
    OFF = 'OFF'
    GPU = 'GPU'
    CPU = 'CPU'

    def __str__(self):
        return self.value
