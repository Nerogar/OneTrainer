from enum import Enum


class DistillationLossType(Enum):
    MSE = 'MSE'
    MAE = 'MAE'
    HUBER = 'HUBER'
    KL_DIVERGENCE = 'KL_DIVERGENCE'

    def __str__(self):
        return self.value
