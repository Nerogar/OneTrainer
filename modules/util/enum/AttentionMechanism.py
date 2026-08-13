from enum import Enum


class AttentionMechanism(Enum):
    SDP = 'SDP'
    FLASH = 'FLASH'
    CUDNN = 'CUDNN'
    FLEX = 'FLEX'

    def __str__(self):
        return self.value
