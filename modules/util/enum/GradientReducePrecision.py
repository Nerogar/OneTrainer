from enum import Enum

import torch


class GradientReducePrecision(Enum):
    WEIGHT_DTYPE = 'WEIGHT_DTYPE'
    FLOAT_32 = 'FLOAT_32'
    WEIGHT_DTYPE_STOCHASTIC = 'WEIGHT_DTYPE_STOCHASTIC'
    FLOAT_32_STOCHASTIC = 'FLOAT_32_STOCHASTIC'

    def torch_dtype(self, weight_dtype: torch.dtype) -> torch.dtype:
        match self:
            case GradientReducePrecision.WEIGHT_DTYPE:
                return weight_dtype
            case GradientReducePrecision.FLOAT_32:
                return torch.float32
            case GradientReducePrecision.WEIGHT_DTYPE_STOCHASTIC:
                return weight_dtype
            case GradientReducePrecision.FLOAT_32_STOCHASTIC:
                return torch.float32
            case _:
                raise ValueError

    def stochastic_rounding(self, weight_dtype: torch.dtype) -> bool:
        match self:
            case GradientReducePrecision.WEIGHT_DTYPE:
                return False
            case GradientReducePrecision.FLOAT_32:
                return False
            case GradientReducePrecision.WEIGHT_DTYPE_STOCHASTIC:
                return weight_dtype == torch.bfloat16
            case GradientReducePrecision.FLOAT_32_STOCHASTIC:
                return weight_dtype == torch.bfloat16
            case _:
                raise ValueError

    def __str__(self):
        return self.value
