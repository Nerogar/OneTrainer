from modules.util.enum.BaseEnum import BaseEnum


class TimestepDistribution(BaseEnum):
    UNIFORM = 'UNIFORM'
    SIGMOID = 'SIGMOID'
    LOGIT_NORMAL = 'LOGIT_NORMAL'
    HEAVY_TAIL = 'HEAVY_TAIL'
    COS_MAP = 'COS_MAP'
    INVERTED_PARABOLA = 'INVERTED_PARABOLA'
