
from modules.module.quantized.LinearW8A8 import (
    LinearFp8Function,
    LinearInt8Function,
    quantize_fp8_tensorwise,
    quantize_int8_tensorwise,
)

import torch

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor


class LinearGGUFA8(GGUFLinear):
    def __init__(self, dtype, compute_dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype
        self._compute_dtype = compute_dtype


    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        x = x_orig.to(self._compute_dtype).reshape(-1, x_orig.shape[-1])
        w = dequantize_gguf_tensor(self.weight)

        if x.shape[0] > 16:
            if self._dtype == torch.int8:
                #TODO tokenwise instead? Higher quality, but requires quantization on forward and backward
                q, q_scale = quantize_int8_tensorwise(w)
                y = LinearInt8Function.apply(x, q, q_scale, self.bias)
            else:
                q, q_scale = quantize_fp8_tensorwise(w)
                y = LinearFp8Function.apply(x, q, q_scale, self.bias)
        else:
            y = torch.nn.functional.linear(x, w, self.bias.to(self._compute_dtype))

        assert y.dtype == self._compute_dtype
        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
