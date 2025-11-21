from enum import StrEnum


class TensorboardMode(StrEnum):
    OFF = "OFF"
    TRAIN_ONLY = "TRAIN_ONLY"
    ALWAYS_ON = "ALWAYS_ON"
