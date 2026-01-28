from enum import Enum


class AttentionMechanism(Enum):
    SDP = 'SDP'
    FLASH = 'FLASH'

    def __str__(self):
        return self.value
