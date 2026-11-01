from enum import Enum


class RunNameMode(Enum):
    DEFAULT = 'DEFAULT'
    FRIENDLY = 'FRIENDLY'
    CUSTOM = 'CUSTOM'

    def __str__(self):
        return self.value
