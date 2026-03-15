from enum import Enum


class PooledOutputHandling(Enum):
    FIRST = 'FIRST'
    LAST = 'LAST'
    AVERAGE = 'AVERAGE'

    def __str__(self):
        return self.value
