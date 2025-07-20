from enum import StrEnum


class TensorboardMode(StrEnum):
    OFF = "off"
    TRAIN_ONLY = "train_only"
    ALWAYS_ON = "always_on"
