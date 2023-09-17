from enum import Enum

import torch


class DataType(Enum):
    NONE = 'NONE'
    FLOAT_16 = 'FLOAT_16'
    FLOAT_32 = 'FLOAT_32'
    BFLOAT_16 = 'BFLOAT_16'
    TFLOAT_32 = 'TFLOAT_32'

    def __str__(self):
        return self.value

    def torch_dtype(self):
        match self:
            case DataType.FLOAT_16:
                return torch.float16
            case DataType.FLOAT_32:
                return torch.float32
            case DataType.BFLOAT_16:
                return torch.bfloat16
            case DataType.TFLOAT_32:
                return torch.float32
            case _:
                return None

    def enable_tf(self):
        return self == DataType.TFLOAT_32

    def enable_loss_scaling(self, weight_dtypes: list['DataType']):
        return self == DataType.FLOAT_16 and all(dtype == DataType.FLOAT_32 for dtype in weight_dtypes)
