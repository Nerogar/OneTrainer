from enum import Enum


class GenerateCaptionsModel(Enum):
    MOONDREAM2 = 'MOONDREAM2'
    BLIP2 = 'BLIP2'
    WD14_VIT_2 = 'WD14_VIT_2'
    WD_EVA02_LARGE_V3 = 'WD_EVA02_LARGE_V3'
    WD_SWINV2_V3 = 'WD_SWINV2_V3'
    JOYTAG = 'JOYTAG'


    def __str__(self):
        return self.value
