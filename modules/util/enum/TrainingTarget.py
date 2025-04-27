from enum import Enum


class TrainingTarget(Enum):
    SAMPLE = 'SAMPLE'
    PRIOR_PREDICTION = 'PRIOR_PREDICTION'

    def __str__(self):
        return self.value
