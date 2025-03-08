from enum import Enum


class GenerateCaptionsModel(Enum):
    BLIP = 'BLIP'
    BLIP2 = 'BLIP2'
    WD14_VIT_2 = 'WD14_VIT_2'
    WD14_SWINV2_v3 = 'WD14_SWINV2_v3'
    WD14_EVA02_v3 = 'WD14_EVA02_v3'

    def __str__(self):
        return self.value
