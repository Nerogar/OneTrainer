from enum import Enum

import torch


class DataType(Enum):
    NONE = 'NONE'
    FLOAT_8 = 'FLOAT_8'
    FLOAT_16 = 'FLOAT_16'
    FLOAT_32 = 'FLOAT_32'
    BFLOAT_16 = 'BFLOAT_16'
    TFLOAT_32 = 'TFLOAT_32'
    INT_8 = 'INT_8'
    NFLOAT_4 = 'NFLOAT_4'
    FLOAT_W8A8 = 'FLOAT_W8A8'
    INT_W8A8 = 'INT_W8A8'
    FLOAT_8_SVD = 'FLOAT_8_SVD'
    NFLOAT_4_SVD = 'NFLOAT_4_SVD'
    FLOAT_W8A8_SVD = 'FLOAT_W8A8_SVD'
    INT_W8A8_SVD = 'INT_W8A8_SVD'
    GGUF = 'GGUF'

    def __str__(self):
        return self.value

    def torch_dtype(
            self,
            supports_quantization: bool = True,
    ):
        if self.is_quantized() and not supports_quantization:
            return torch.float16

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

    def is_quantized(self):
        return self in [DataType.FLOAT_8,
                        DataType.INT_8,
                        DataType.FLOAT_W8A8,
                        DataType.INT_W8A8,
                        DataType.NFLOAT_4,
                        DataType.FLOAT_8_SVD,
                        DataType.FLOAT_W8A8_SVD,
                        DataType.INT_W8A8_SVD,
                        DataType.NFLOAT_4_SVD]

    def quantize_fp8(self):
        return self == DataType.FLOAT_8 or self == DataType.FLOAT_8_SVD

    def quantize_int8(self):
        return self == DataType.INT_8

    def quantize_fpW8A8(self):
        return self == DataType.FLOAT_W8A8 or self == DataType.FLOAT_W8A8_SVD

    def quantize_intW8A8(self):
        return self == DataType.INT_W8A8 or self == DataType.INT_W8A8_SVD

    def quantize_nf4(self):
        return self == DataType.NFLOAT_4 or self == DataType.NFLOAT_4_SVD

    def quantize_svd(self):
        return self in [DataType.FLOAT_8_SVD,
                        DataType.NFLOAT_4_SVD,
                        DataType.FLOAT_W8A8_SVD,
                        DataType.INT_W8A8_SVD]
