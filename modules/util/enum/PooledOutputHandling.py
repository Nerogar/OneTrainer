from enum import Enum


class PooledOutputHandling(Enum):
    FIRST = 'FIRST'
    AVERAGE = 'AVERAGE'

    def __str__(self):
        return self.value
