from modules.module.quantized.LinearA8 import (
    fp8_backward_act_axiswise,
    fp8_forward_axiswise,
    int8_backward_act_axiswise,
    int8_forward_axiswise,
)

import torch
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

import gguf

UNQUANTIZED_TYPES = [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]

class LinearGGUFIntA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        #axiswise performs better than tensorwise in tests, even though
        #it requires another requant during backward - but requant is cheap
        return int8_forward_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return int8_backward_act_axiswise(output, weight), None, None

class LinearGGUFFpA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        return fp8_forward_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return fp8_backward_act_axiswise(output, weight), None, None

class LinearGGUFA8(GGUFLinear):
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        x = x_orig.to(self.compute_dtype).reshape(-1, x_orig.shape[-1])
        w = dequantize_gguf_tensor(self.weight).detach()

        if x.shape[0] > 16 and self.weight.quant_type not in UNQUANTIZED_TYPES:
            if self._dtype == torch.int8:
                y = LinearGGUFIntA8RequantFunction.apply(x, w, self.bias)
            else:
                y = LinearGGUFFpA8RequantFunction.apply(x, w, self.bias)
        else:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
            bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
            y = torch.nn.functional.linear(x, w, bias)

        assert y.dtype == self.compute_dtype
        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
