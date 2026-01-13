from enum import Enum


class AttentionMechanism(Enum):
    SDP = 'SDP'
    FLASH = 'FLASH'
    SPLIT = 'SPLIT'
    FLASH_SPLIT = 'FLASH_SPLIT'

    def __str__(self):
        return self.value
