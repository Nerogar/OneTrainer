from enum import Enum


class PathIOType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def __str__(self):
        return self.value
