from enum import Enum


class DistillationTargetMode(Enum):
    RAW = 'RAW'
    SCALED_LOSS_WEIGHT = 'SCALED_LOSS_WEIGHT'
    CFG_DISTILL = 'CFG_DISTILL'
    STEP_ROLLOUT = 'STEP_ROLLOUT'

    def __str__(self):
        return self.value