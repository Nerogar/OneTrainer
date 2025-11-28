from modules.util.enum.BaseEnum import BaseEnum

import torch


class DataType(BaseEnum):
    NONE = 'NONE'
    FLOAT_8 = 'FLOAT_8'
    FLOAT_16 = 'FLOAT_16'
    FLOAT_32 = 'FLOAT_32'
    BFLOAT_16 = 'BFLOAT_16'
    TFLOAT_32 = 'TFLOAT_32'
    INT_8 = 'INT_8'
    NFLOAT_4 = 'NFLOAT_4'
    GGUF = 'GGUF'

    def pretty_print(self):
        return {
            DataType.NONE: '',
            DataType.FLOAT_8: 'Float8',
            DataType.FLOAT_16: 'Float16',
            DataType.FLOAT_32: 'Float32',
            DataType.BFLOAT_16: 'BFloat16',
            DataType.TFLOAT_32: 'TFloat32',
            DataType.INT_8: 'Int8',
            DataType.NFLOAT_4: 'NFloat4',
            DataType.GGUF: 'GGUF'
        }[self]

    @staticmethod
    def is_enabled(value, context=None):
        if context == "embeddings" or context == "lora":
            return value in [DataType.FLOAT_32, DataType.BFLOAT_16]
        elif context == "convert_window":
            return value in [DataType.FLOAT_32, DataType.FLOAT_16, DataType.BFLOAT_16]
        elif context == "training_dtype":
            return value in [DataType.FLOAT_32, DataType.FLOAT_16, DataType.BFLOAT_16, DataType.TFLOAT_32]
        elif context == "training_fallback":
            return value in [DataType.FLOAT_32, DataType.BFLOAT_16]
        elif context == "output_dtype":
            return value in [
                DataType.FLOAT_16,
                DataType.FLOAT_32,
                DataType.BFLOAT_16,
                DataType.FLOAT_8,
                DataType.NFLOAT_4
            ]
        elif context == "transformer_dtype":
            return value in [
                DataType.FLOAT_32,
                DataType.BFLOAT_16,
                DataType.FLOAT_16,
                DataType.FLOAT_8,
                # DataType.INT_8,  # TODO: reactivate when the int8 implementation is fixed in bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1332
                DataType.NFLOAT_4,
                DataType.GGUF
            ]
        else: # model_dtypes
            return value in [
                DataType.FLOAT_32,
                DataType.BFLOAT_16,
                DataType.FLOAT_16,
                DataType.FLOAT_8,
                # DataType.INT_8,  # TODO: reactivate when the int8 implementation is fixed in bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1332
                DataType.NFLOAT_4,
            ]

        return True

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
                        DataType.NFLOAT_4]

    def quantize_fp8(self):
        return self == DataType.FLOAT_8

    def quantize_int8(self):
        return self == DataType.INT_8

    def quantize_nf4(self):
        return self == DataType.NFLOAT_4
