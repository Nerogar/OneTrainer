from enum import Enum


class DPORefMode(Enum):
    NEW_ADAPTER = "NEW_ADAPTER"
    EXISTING_ADAPTER = "EXISTING_ADAPTER"

    def __str__(self):
        return self.value
