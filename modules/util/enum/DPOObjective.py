from enum import Enum


class DPOObjective(Enum):
    SIGMOID = 'SIGMOID'
    IPO = 'IPO'

    def __str__(self):
        return self.value
