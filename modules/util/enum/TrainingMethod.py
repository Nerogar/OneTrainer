from modules.util.enum.BaseEnum import BaseEnum


class TrainingMethod(BaseEnum):
    FINE_TUNE = 'FINE_TUNE'
    LORA = 'LORA'
    EMBEDDING = 'EMBEDDING'
    FINE_TUNE_VAE = 'FINE_TUNE_VAE'

    def pretty_print(self):
        return {
            TrainingMethod.FINE_TUNE: "Fine Tune",
            TrainingMethod.LORA: "Lora",
            TrainingMethod.EMBEDDING: "Embedding",
            TrainingMethod.FINE_TUNE_VAE: "Fine Tune VAE",
        }[self]

    @staticmethod
    def is_enabled(value, context=None):
        # TODO
        if context == "convert_window":
            return value in [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING]
        else: # Main window
            pass

        return True
