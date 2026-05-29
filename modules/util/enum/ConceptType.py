from enum import Enum


class ConceptType(Enum):
    STANDARD = 'STANDARD'
    VALIDATION = 'VALIDATION'
    PRIOR_PREDICTION = 'PRIOR_PREDICTION'
    DPO_CHOSEN = 'DPO_CHOSEN'
    DPO_REJECTED = 'DPO_REJECTED'
    DPO_CHOSEN_VAL = 'DPO_CHOSEN_VAL'
    DPO_REJECTED_VAL = 'DPO_REJECTED_VAL'

    def __str__(self):
        return self.value
