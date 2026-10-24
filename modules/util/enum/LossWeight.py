from enum import Enum


class LossWeight(Enum):
    CONSTANT = 'CONSTANT'
    P2 = 'P2'
    MIN_SNR_GAMMA = 'MIN_SNR_GAMMA'
    DEBIASED_ESTIMATION = 'DEBIASED_ESTIMATION'
    SIGMA = 'SIGMA'

    def supports_flow_matching(self) -> bool:
        return self == LossWeight.CONSTANT \
            or self == LossWeight.SIGMA

    def __str__(self):
        return self.value
