from enum import Enum


class SamplingMethod(Enum):
    STANDARD = 'STANDARD'  # the model's own way of generating a sample, with nothing swapped in
    HANDOFF_LOW_NOISE = 'HANDOFF_LOW_NOISE'  # the trained transformer denoises down to the expert's first sigma, the expert runs the tail
    DISTILLED = 'DISTILLED'  # the expert runs its own full schedule from noise; the trained transformer never runs

    def __str__(self):
        return self.value
