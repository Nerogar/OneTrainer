from enum import Enum


class AttentionMechanism(Enum):
    DEFAULT = 'DEFAULT'
    XFORMERS = 'XFORMERS'
    SDP = 'SDP'

    def __str__(self):
        return self.value
