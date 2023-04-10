from enum import Enum


class TrainingMethod(Enum):
    FINE_TUNE = 'FINE_TUNE'
    LORA = 'LORA'
    EMBEDDING = 'EMBEDDING'

    def __str__(self):
        return self.value
