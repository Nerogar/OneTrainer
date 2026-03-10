from enum import Enum


class DistillationTargetMode(Enum):
    RAW = 'RAW'
    CFG_SCALE = 'CFG_SCALE'
    STEP_ROLLOUT = 'STEP_ROLLOUT'

    def __str__(self):
        return self.value