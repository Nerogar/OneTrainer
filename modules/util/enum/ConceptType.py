from modules.util.enum.BaseEnum import BaseEnum


class ConceptType(BaseEnum):
    ALL = 'ALL'
    STANDARD = 'STANDARD'
    VALIDATION = 'VALIDATION'
    PRIOR_PREDICTION = 'PRIOR_PREDICTION'

    @staticmethod
    def is_enabled(value, context=None):
        if context == "all":
            return True
        elif context == "prior_pred_enabled":
            return value in [ConceptType.STANDARD, ConceptType.VALIDATION, ConceptType.PRIOR_PREDICTION]
        else: # prior_pred_disabled
            return value in [ConceptType.STANDARD, ConceptType.VALIDATION]
