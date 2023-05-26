from enum import Enum


class Optimizer(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'

    def __str__(self):
        return self.value
