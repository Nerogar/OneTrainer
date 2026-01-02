try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class TensorboardMode(StrEnum):
    OFF = "OFF"
    TRAIN_ONLY = "TRAIN_ONLY"
    ALWAYS_ON = "ALWAYS_ON"
