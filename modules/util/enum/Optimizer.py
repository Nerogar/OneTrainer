from enum import Enum


class Optimizer(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'
    ADAM_8BIT = 'ADAM_8BIT'
    ADAMW_8BIT = 'ADAMW_8BIT'

    def __str__(self):
        return self.value
