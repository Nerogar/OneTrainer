from enum import Enum


class FileType(Enum):
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'
    AUDIO = 'AUDIO'

    def __str__(self):
        return self.value
