from modules.module.quantized.LinearW8A8 import quantize_fp8_axiswise, quantize_int8_axiswise, fp8_forward_tokenwise, int8_forward_tokenwise, int8_backward_W_tensorwise_A_axiswise, fp8_backward_W_tensorwise_A_axiswise
import torch
from torch import Tensor

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor


class LinearGGUFIntA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        #axiswise performs better than tensorwise in tests, even though
        #it requires another requant during backward - but requant is cheap
        weight_q, weight_scale = quantize_int8_axiswise(weight, dim=-1)
        return int8_forward_tokenwise(x, weight_q, weight_scale.T, bias)

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        weight_q, weight_scale = quantize_int8_axiswise(weight, dim=0)
        return int8_backward_W_tensorwise_A_axiswise(x, weight_q, weight_scale), None, None

class LinearGGUFFpA8RequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight)
        weight_q, weight_scale = quantize_fp8_axiswise(weight, dim=-1)
        return fp8_forward_tokenwise(x, weight_q, weight_scale, bias)

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False):
            raise NotImplementedError("GGUF cannot be used for full finetuning")
        weight, = ctx.saved_tensors
        weight_q, weight_scale = quantize_fp8_axiswise(weight, dim=0)
        return fp8_backward_W_tensorwise_A_axiswise(x, weight_q, weight_scale), None, None


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
