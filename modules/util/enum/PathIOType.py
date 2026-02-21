from enum import Enum


class PathIOType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MODEL = "MODEL"

    def __str__(self):
        return self.value
