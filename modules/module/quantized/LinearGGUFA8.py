from modules.module.quantized.mixin.LoRAFusableLinearMixin import LoRAFusableLinearMixin
from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    quantize_axiswise,
    quantize_fp8_axiswise,
    quantize_int8_axiswise,
)

import torch
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

import gguf

UNQUANTIZED_TYPES = [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]


#unlike LinearW8A8, whose weight is quantized once and tensorwise, the GGUF weight is dequantized and
#requantized axiswise per pass (it can be, since the dequant happens anyway). That leaves two scale
#vectors varying along different axes - per-token on M and per-output-channel on N - which cannot be
#collapsed into one, so this layer uses the rowcol_ variants of the epilogue-scaled kernels.

#the weight reaches these dequantized, so unlike LinearW8A8 they are told which 8-bit dtype to
#requantize to. Only the torch forward differs beyond the quantizer: int8 and fp8 have separate
#torch mms with separate scaling
@torch.no_grad()
def forward_axiswise_postscaled_torch(dtype: torch.dtype, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    if dtype == torch.int8:
        x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
        w_8, w_scale = quantize_int8_axiswise(weight, dim=-1)
        res = torch._int_mm(x_8, w_8.T)
        res_scaled = res.float().mul_(w_scale.T).mul_(x_scale).to(compute_dtype)
    else:
        x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
        w_8, w_scale = quantize_fp8_axiswise(weight, dim=-1)
        one = torch.ones(1, device=x.device)
        res = torch._scaled_mm(x_8, w_8.T, scale_a=one, scale_b=one, out_dtype=torch.float)
        res_scaled = res.mul_(w_scale.T).mul_(x_scale).to(compute_dtype) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def forward_axiswise_epiloguescaled_triton(dtype: torch.dtype, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_axiswise(x, dim=-1, dtype=dtype)
    w_8, w_scale = quantize_axiswise(weight, dim=-1, dtype=dtype)
    #the mm folds the per-token scale (axis 0) and the per-channel weight scale
    #(axis 1) into the epilogue and returns compute_dtype directly
    res_scaled = mm_8bit(x_8, w_8.T, out_dtype=compute_dtype, scale_m=x_scale, scale_n=w_scale)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def backward_axiswise_epiloguescaled_triton(dtype: torch.dtype, output: Tensor, weight: Tensor) -> Tensor:
    output_8, output_scale = quantize_axiswise(output, dim=-1, dtype=dtype)
    w_8, w_scale = quantize_axiswise(weight, dim=0, dtype=dtype)
    return mm_8bit(output_8.contiguous(), w_8, out_dtype=output.dtype, scale_m=output_scale, scale_n=w_scale)


@torch.no_grad()
def forward_axiswise_lora_epiloguescaled_triton(dtype: torch.dtype, x: Tensor, weight: Tensor, bias: Tensor | None, compute_dtype: torch.dtype, x_down: Tensor, lora_up: Tensor) -> Tensor:
    x_8, x_scale = quantize_axiswise(x, dim=-1, dtype=dtype)
    w_8, w_scale = quantize_axiswise(weight, dim=-1, dtype=dtype)
    res_scaled = mm_8bit(x_8, w_8.T, out_dtype=compute_dtype, scale_m=x_scale, scale_n=w_scale, lora_xd=x_down, lora_up=lora_up)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def backward_axiswise_lora_epiloguescaled_triton(dtype: torch.dtype, output: Tensor, weight: Tensor, grad_x_down_pre: Tensor, lora_down: Tensor) -> Tensor:
    output_8, output_scale = quantize_axiswise(output, dim=-1, dtype=dtype)
    w_8, w_scale = quantize_axiswise(weight, dim=0, dtype=dtype)
    return mm_8bit(output_8.contiguous(), w_8, out_dtype=output.dtype, scale_m=output_scale, scale_n=w_scale, lora_xd=grad_x_down_pre, lora_up=lora_down.to(grad_x_down_pre.dtype))


forward_axiswise = forward_axiswise_epiloguescaled_triton
backward_axiswise = backward_axiswise_epiloguescaled_triton
forward_axiswise_lora = forward_axiswise_lora_epiloguescaled_triton
backward_axiswise_lora = backward_axiswise_lora_epiloguescaled_triton


class LinearGGUFA8RequantFunction(torch.autograd.Function):
    #`weight` is the dequantized GGUF weight, so saving it lets backward requantize instead of
    #dequantizing the GGUF blocks a second time
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, dtype: torch.dtype, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight)
        ctx.dtype = dtype
        return forward_axiswise(dtype, x, weight, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")

        weight, = ctx.saved_tensors
        return backward_axiswise(ctx.dtype, output, weight), None, None, None, None


class LinearGGUFA8RequantLoRAFunction(torch.autograd.Function):
    #LinearGGUFA8RequantFunction plus a low-rank update: it owns the down-projection and the
    #dropout, so the LoRA dgrad folds into the backward epilogue and the (M, out) product is
    #never materialized
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, dtype: torch.dtype, bias: Tensor | None, compute_dtype: torch.dtype, lora_down: Tensor, lora_up: Tensor, dropout_mask: Tensor | None) -> Tensor:
        x_down = torch.nn.functional.linear(x, lora_down)
        if dropout_mask is not None:
            x_down = x_down * dropout_mask
        ctx.dtype = dtype
        ctx.save_for_backward(weight, x.to(x_down.dtype), lora_down, lora_up, dropout_mask, x_down)
        return forward_axiswise_lora(dtype, x, weight, bias, compute_dtype, x_down, lora_up)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.needs_input_grad[1:5] != (False, False, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")

        weight, x, lora_down, lora_up, dropout_mask, x_down = ctx.saved_tensors
        needs_x, needs_down, needs_up = ctx.needs_input_grad[0], ctx.needs_input_grad[5], ctx.needs_input_grad[6]
        #grad_x_down_pre (post-dropout-backward) feeds both the grad_x epilogue fold and grad_lora_down
        grad_x_down_pre = None
        if needs_x or needs_down:
            grad_x_down_pre = grad_output @ lora_up.T
            if dropout_mask is not None:
                grad_x_down_pre = grad_x_down_pre * dropout_mask
        grad_x = backward_axiswise_lora(ctx.dtype, grad_output, weight, grad_x_down_pre, lora_down) if needs_x else None
        grad_lora_down = (grad_x_down_pre.T @ x).to(lora_down.dtype) if needs_down else None
        grad_lora_up = x_down.T @ grad_output if needs_up else None
        return grad_x, None, None, None, None, grad_lora_down, grad_lora_up, None


class LinearGGUFA8(
    GGUFLinear,
    LoRAFusableLinearMixin,
):
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        x = x_orig.reshape(-1, x_orig.shape[-1])

        w = dequantize_gguf_tensor(self.weight.detach())
        #the 8-bit path only pays for itself above a few tokens, and an unquantized GGUF type has
        #nothing to requantize - dequantize_gguf_tensor already hands back the dense weight
        if x.shape[0] > 16 and self.weight.quant_type not in UNQUANTIZED_TYPES:
            #axiswise performs better than tensorwise in tests, even though
            #it requires another requant during backward - but requant is cheap
            y = LinearGGUFA8RequantFunction.apply(x, w, self._dtype, self.bias, self.compute_dtype)
        else:
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

    def forward_with_lora(self, x_orig: torch.Tensor, lora_down: torch.Tensor, lora_up: torch.Tensor, dropout: torch.nn.Dropout, alpha: torch.Tensor) -> torch.Tensor:
        lora_rank = lora_down.shape[0]

        x = x_orig.reshape(-1, x_orig.shape[-1])
        if x.shape[0] <= 16 or self.weight.quant_type in UNQUANTIZED_TYPES:
            ld = torch.nn.functional.linear(dropout(torch.nn.functional.linear(x_orig, lora_down)), lora_up)
            return LinearGGUFA8.forward(self, x_orig) + ld * (alpha / lora_rank)

        #the cast matches lora_up (kept in its own dtype, e.g. f32) to the down-projection's autocast
        #dtype, which the epilogue dot needs; dropout(ones) yields the scaled 0/(1/(1-p)) mask, drawn
        #here rather than inside LinearGGUFA8RequantLoRAFunction so the RNG stays inductor-native
        lora_up_scaled = (lora_up * (alpha / lora_rank)).T.to(self.compute_dtype)
        dropout_mask = dropout(torch.ones(x.shape[0], lora_rank, device=x.device, dtype=self.compute_dtype)) if (dropout.training and dropout.p > 0) else None
        y = self._fused_lora_forward(x, lora_down, lora_up_scaled, dropout_mask)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

    #see LinearW8A8._fused_lora_forward for what the fused node buys and what the operands are
    def _fused_lora_forward(self, x: Tensor, lora_down: Tensor, lora_up: Tensor, dropout_mask: Tensor | None) -> Tensor:
        assert not self.weight.requires_grad
        w = dequantize_gguf_tensor(self.weight.detach())
        return LinearGGUFA8RequantLoRAFunction.apply(x, w, self._dtype, self.bias, self.compute_dtype, lora_down, lora_up, dropout_mask)
