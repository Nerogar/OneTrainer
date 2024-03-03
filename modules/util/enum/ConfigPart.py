from enum import Enum


class ConfigPart(Enum):
    NONE = 'NONE'
    SETTINGS = 'SETTINGS'
    ALL = 'ALL'

    def __str__(self):
        return self.value
