from enum import Enum


class Optimizer(Enum):
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'

    def __str__(self):
        return self.value
