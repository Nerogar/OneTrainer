from enum import Enum


class DPOPatienceMode(Enum):
    EITHER = "EITHER"
    BOTH = "BOTH"

    def __str__(self):
        return self.value
