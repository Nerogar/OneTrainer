from enum import Enum


class CloudAction(Enum):
    NONE = 'NONE'
    STOP = 'STOP'
    DELETE = 'DELETE'
    def __str__(self):
        return self.value
