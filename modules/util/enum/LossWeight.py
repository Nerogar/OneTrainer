from modules.util.enum.BaseEnum import BaseEnum


class LossWeight(BaseEnum):
    CONSTANT = 'CONSTANT'
    P2 = 'P2'
    MIN_SNR_GAMMA = 'MIN_SNR_GAMMA'
    DEBIASED_ESTIMATION = 'DEBIASED_ESTIMATION'
    SIGMA = 'SIGMA'

    def supports_flow_matching(self) -> bool:
        return self == LossWeight.CONSTANT \
            or self == LossWeight.SIGMA

    def pretty_print(self):
        return {
            LossWeight.CONSTANT: 'Constant',
            LossWeight.P2: 'P2',
            LossWeight.MIN_SNR_GAMMA: 'Min SNR Gamma',
            LossWeight.DEBIASED_ESTIMATION: 'Debiased Estimation',
            LossWeight.SIGMA: 'Sigma',
        }[self]

    @staticmethod
    def is_enabled(value, context=None):
        if context == "flow_matching":
            return value in [LossWeight.CONSTANT, LossWeight.SIGMA]
        else:
            return value in [LossWeight.CONSTANT, LossWeight.P2, LossWeight.MIN_SNR_GAMMA, LossWeight.DEBIASED_ESTIMATION]
