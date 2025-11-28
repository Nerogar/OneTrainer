import modules.util.multi_gpu_util as multi
from modules.util.enum.BaseEnum import BaseEnum


class LossScaler(BaseEnum):
    NONE = 'NONE'
    BATCH = 'BATCH'
    GLOBAL_BATCH = 'GLOBAL_BATCH'
    GRADIENT_ACCUMULATION = 'GRADIENT_ACCUMULATION'
    BOTH = 'BOTH'
    GLOBAL_BOTH = 'GLOBAL_BOTH'

    def get_scale(self, batch_size: int, accumulation_steps: int) -> int:
        match self:
            case LossScaler.NONE:
                return 1
            case LossScaler.BATCH:
                return batch_size
            case LossScaler.GLOBAL_BATCH:
                return batch_size * multi.world_size()
            case LossScaler.GRADIENT_ACCUMULATION:
                return accumulation_steps
            case LossScaler.BOTH:
                return accumulation_steps * batch_size
            case LossScaler.GLOBAL_BOTH:
                return accumulation_steps * batch_size * multi.world_size()
            case _:
                raise ValueError
