from enum import Enum


class LossFunction(Enum):
    MSE = 'MSE'

    def __str__(self):
        return self.value
