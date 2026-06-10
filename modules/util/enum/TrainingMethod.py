from enum import Enum


class TrainingMethod(Enum):
    FINE_TUNE = 'FINE_TUNE'
    LORA = 'LORA'
    EMBEDDING = 'EMBEDDING'
    FINE_TUNE_VAE = 'FINE_TUNE_VAE'

    def __str__(self):
        return self.value
