from modules.module.quantized.LinearW8A8 import (
    quantize_fp8_axiswise,
    quantize_int8_axiswise,
)
from modules.util.triton_mm_8bit import mm_8bit as triton_mm_8bit

import torch
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor


def int8_forward_both_axiswise(x: Tensor, weight: Tensor, bias: Tensor=None) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=-1)
    res = torch._int_mm(x_8, w_8.T)
    res_scaled = res.to(x.dtype).mul_(w_scale.T).mul_(x_scale)
    if bias is not None:
        res_scaled.add_(bias.to(x.dtype))
    return res_scaled

def fp8_forward_both_axiswise(x: Tensor, weight: Tensor, bias: Tensor=None) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=-1)
    one = torch.ones(1, device=x.device)
    res = torch._scaled_mm(x_8, w_8.T, scale_a=one, scale_b=one, out_dtype=x.dtype)
    res_scaled = res.mul_(w_scale.T).mul_(x_scale) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias.to(x.dtype))
    return res_scaled

def int8_backward_both_axiswise(x: Tensor, weight: Tensor) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=0)
    mm_res = triton_mm_8bit(x_8, w_8)
    return mm_res.to(x.dtype).mul_(w_scale).mul_(x_scale)

def fp8_backward_both_axiswise(x: Tensor, weight: Tensor) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=0)
    mm_res = triton_mm_8bit(x_8, w_8)
    return mm_res.to(x.dtype).mul_(w_scale).mul_(x_scale)

class LinearGGUFIntA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        #axiswise performs better than tensorwise in tests, even though
        #it requires another requant during backward - but requant is cheap
        return int8_forward_both_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return int8_backward_both_axiswise(x, weight), None, None

class LinearGGUFFpA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        return fp8_forward_both_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return fp8_backward_both_axiswise(x, weight), None, None


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
                y = LinearGGUFIntA8RequantFunction.apply(x, w, self.bias)
            else:
                y = LinearGGUFFpA8RequantFunction.apply(x, w, self.bias)
        else:
            y = torch.nn.functional.linear(x, w, self.bias.to(self._compute_dtype))

        assert y.dtype == self._compute_dtype
        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
