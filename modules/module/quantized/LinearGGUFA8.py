from modules.module.quantized.mixin.CompressedWeightMixin import CompressedWeightMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    quantize_fp8_axiswise,
    quantize_int8_axiswise,
)

import torch
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

import gguf

UNQUANTIZED_TYPES = [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]


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

@torch.no_grad()
def int8_backward_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_int8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_int8_axiswise(weight, dim=0)
    mm_res = mm_8bit(output_8.contiguous(), w_8)
    return mm_res.float().mul_(w_scale).mul_(output_scale).to(output.dtype)

@torch.no_grad()
def fp8_backward_axiswise(output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_fp8_axiswise(output, dim=-1)
    w_8, w_scale = quantize_fp8_axiswise(weight, dim=0)
    mm_res = mm_8bit(output_8.contiguous(), w_8)
    return mm_res.float().mul_(w_scale).mul_(output_scale).to(output.dtype)

class LinearGGUFIntA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight)
        #axiswise performs better than tensorwise in tests, even though
        #it requires another requant during backward - but requant is cheap
        return int8_forward_axiswise(x, weight, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return int8_backward_axiswise(output, weight), None, None, None

class LinearGGUFFpA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight)
        return fp8_forward_axiswise(x, weight, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        return fp8_backward_axiswise(output, weight), None, None, None

class LinearGGUFA8(
    GGUFLinear,
    QuantizedModuleMixin,
    CompressedWeightMixin,
):
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype
        self._quant_type = None

        self._init_compressed_state()

    @torch.no_grad()
    def quantize(self, device: torch.device | None = None):
        # the GGUF weight is already quantized (loaded pre-packed); "quantizing" this layer just
        # means storing its packed bytes nvCOMP-compressed. quant_type is not recoverable from the
        # raw blob, so capture it before _compress_weight() replaces the stored weight.
        if not self.compress:
            return
        self._quant_type = self.weight.quant_type
        self._compress_weight(device=device)

    def _dequantized_weight(self) -> torch.Tensor:
        if self._compressed:
            # _decompress reinterprets the blob back to the packed bytes; re-wrap as a GGUFParameter
            # so dequantize_gguf_tensor sees the quant_type again
            packed = GGUFParameter(self._decompress(self.weight.detach().as_tensor()), quant_type=self._quant_type)
            return dequantize_gguf_tensor(packed)
        return dequantize_gguf_tensor(self.weight.detach())

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        x = x_orig.reshape(-1, x_orig.shape[-1])
        w = self._dequantized_weight()
        quant_type = self._quant_type if self._compressed else self.weight.quant_type

        if x.shape[0] > 16 and quant_type not in UNQUANTIZED_TYPES:
            if self._dtype == torch.int8:
                y = LinearGGUFIntA8RequantFunction.apply(x, w, self.bias, self.compute_dtype)
            else:
                y = LinearGGUFFpA8RequantFunction.apply(x, w, self.bias, self.compute_dtype)
        else:
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))
