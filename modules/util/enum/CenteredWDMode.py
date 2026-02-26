from enum import Enum


class CenteredWDMode(str, Enum):
    FULL = "full"
    FLOAT8 = "float8"
    INT8 = "int8"
    INT4 = "int4"

    def __str__(self):
        return self.name