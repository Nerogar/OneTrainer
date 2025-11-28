from enum import Enum


class GenerateCaptionsModel(Enum):
    BLIP = 'BLIP'
    BLIP2 = 'BLIP2'
    WD14_VIT_2 = 'WD14_VIT_2'

    def __str__(self):
        return self.value
