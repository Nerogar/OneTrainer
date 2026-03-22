from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    quantize_fp8_axiswise,
    quantize_int8_axiswise,
)

import torch
from torch import Tensor

#TODO share code

@torch.no_grad()
def int8_forward_axiswise(x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=-1)
    res = torch._int_mm(x_8, w_8.T)
    res_scaled = res.float().mul_(w_scale.T).mul_(x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def fp8_forward_axiswise(x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=-1)
    one = torch.ones(1, device=x.device)
    res = torch._scaled_mm(x_8, w_8.T, scale_a=one, scale_b=one, out_dtype=torch.float)
    res_scaled = res.mul_(w_scale.T).mul_(x_scale).to(compute_dtype) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

def int8_backward_act_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_int8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=0)
    #almost always, grad outputs are already contiguous and this is a no-op. But there are some grad outputs from SDXL that are non-contiguous:
    output_8 = output_8.contiguous()
    mm_res = mm_8bit(output_8, w_8)
    return mm_res.to(output.dtype).mul_(w_scale).mul_(output_scale)

def fp8_backward_act_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_fp8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=0)
    mm_res = mm_8bit(output_8.contiguous(), w_8)
    return mm_res.to(output.dtype).mul_(w_scale).mul_(output_scale)

def int8_backward_weight_axiswise(output: Tensor, x: Tensor) -> Tensor:
    output_8, output_scale = quantize_int8_axiswise(output, dim=0)
    x_8, x_scale = quantize_int8_axiswise(x, dim=0)
    #TODO could be more efficient using a kernel that accepts a non-contiguous lhs matrix
    mm_res = mm_8bit(output_8.T.contiguous(), x_8)
    return mm_res.to(x.dtype).mul_(output_scale.T).mul_(x_scale)

def fp8_backward_weight_axiswise(output: Tensor, x: Tensor) -> Tensor:
    output_8, output_scale = quantize_fp8_axiswise(output, dim=0)
    x_8, x_scale = quantize_fp8_axiswise(x, dim=0)
    mm_res = mm_8bit(output_8.T.contiguous(), x_8)
    return mm_res.to(x.dtype).mul_(output_scale.T).mul_(x_scale)


class LinearW16IntA8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(x, weight)
        return int8_forward_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight = ctx.saved_tensors

        grad_x, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            # grad_output @ weight.T
            grad_x = int8_backward_act_axiswise(grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_output.T @ x
            grad_weight = int8_backward_weight_axiswise(grad_output, x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias

class LinearW16FpA8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(x, weight)
        return fp8_forward_axiswise(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight = ctx.saved_tensors

        grad_x, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            # grad_output @ weight.T
            grad_x = fp8_backward_act_axiswise(grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_output.T @ x
            grad_weight = fp8_backward_weight_axiswise(grad_output, x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias

class LinearW16A8(torch.nn.Linear):
    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x = x_orig.to(self.weight.dtype).reshape(-1, x_orig.shape[-1])
        if x.shape[0] > 16:
            if self._dtype == torch.int8:
                y = LinearW16IntA8Function.apply(x, self.weight, self.bias)
            else:
                y = LinearW16FpA8Function.apply(x, self.weight, self.bias)
            return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
        else:
            return super().forward(x_orig)
