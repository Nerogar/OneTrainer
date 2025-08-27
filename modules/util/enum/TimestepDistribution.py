from enum import Enum


class TimestepDistribution(Enum):
    UNIFORM = 'UNIFORM'
    SIGMOID = 'SIGMOID'
    LOGIT_NORMAL = 'LOGIT_NORMAL'
    HEAVY_TAIL = 'HEAVY_TAIL'
    COS_MAP = 'COS_MAP'
    PARABOLA = 'PARABOLA'

    def __str__(self):
        return self.value
