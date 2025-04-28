from enum import Enum


class ConceptType(Enum):
    STANDARD = 'STANDARD'
    VALIDATION = 'VALIDATION'
    PRIOR_PREDICTION = 'PRIOR_PREDICTION'

    def __str__(self):
        return self.value
