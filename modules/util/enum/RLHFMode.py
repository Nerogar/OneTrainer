from enum import Enum


class RLHFMode(Enum):
    DPO = "DPO"

    def __str__(self):
        return self.value
