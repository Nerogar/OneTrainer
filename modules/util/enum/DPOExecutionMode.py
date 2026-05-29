from enum import Enum


class DPOExecutionMode(Enum):
    SEQUENTIAL = 'SEQUENTIAL'
    POLICY_CONCURRENT = 'POLICY_CONCURRENT'
    FULL_CONCURRENT = 'FULL_CONCURRENT'

    def __str__(self):
        return self.value
