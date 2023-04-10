from enum import Enum


class LossFunction(Enum):
    MSE = 'MSE'
    MASKED_MSE = 'MASKED_MSE'

    def __str__(self):
        return self.value
