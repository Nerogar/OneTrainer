from enum import Enum


class CloudType(Enum):
    RUNPOD = 'RUNPOD'
    LINUX = 'LINUX'
    def __str__(self):
        return self.value
