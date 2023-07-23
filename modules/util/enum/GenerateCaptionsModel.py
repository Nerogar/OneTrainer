from enum import Enum


class GenerateCaptionsModel(Enum):
    BLIP = 'BLIP'
    BLIP2 = 'BLIP2'

    def __str__(self):
        return self.value
