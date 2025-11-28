from modules.util.enum.BaseEnum import BaseEnum


class LearningRateScheduler(BaseEnum):
    CONSTANT = 'CONSTANT'
    LINEAR = 'LINEAR'
    COSINE = 'COSINE'
    COSINE_WITH_RESTARTS = 'COSINE_WITH_RESTARTS'
    COSINE_WITH_HARD_RESTARTS = 'COSINE_WITH_HARD_RESTARTS'
    REX = 'REX'
    ADAFACTOR = 'ADAFACTOR'
    CUSTOM = 'CUSTOM'
