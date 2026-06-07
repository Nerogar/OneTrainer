from enum import Enum


class DistillationCacheMode(Enum):
    DISABLED = 'DISABLED'
    GENERATE_CACHE = 'GENERATE_CACHE'
    USE_CACHE = 'USE_CACHE'

    def __str__(self):
        return self.value
