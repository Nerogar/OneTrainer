
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    dequantize,
    quantize_fp8_axiswise,
    quantize_fp8_tensorwise,
    quantize_int8_axiswise,
    quantize_int8_tensorwise,
)

import torch
from torch import Tensor, nn


@torch.no_grad()
def int8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def fp8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    one = torch.ones(1, device=x.device)
    res = torch._scaled_mm(x_8, weight.T, scale_a=one, scale_b=weight_scale.float(), out_dtype=torch.float)
    res_scaled = res.mul_(x_scale).to(compute_dtype) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def int8_backward_axiswise(output: Tensor, weight: Tensor, weight_scale: float) -> Tensor:
    output_8, output_scale = quantize_int8_axiswise(output, dim=-1)
    #almost always, grad outputs are already contiguous and this is a no-op. But there are some grad outputs from SDXL that are non-contiguous:
    mm_res = mm_8bit(output_8.contiguous(), weight)
    return mm_res.float().mul_(weight_scale * output_scale).to(output.dtype)

@torch.no_grad()
def fp8_backward_axiswise(output: Tensor, weight: Tensor, weight_scale: float) -> Tensor:
    output_8, output_scale = quantize_fp8_axiswise(output, dim=-1)
    mm_res = mm_8bit(output_8.contiguous(), weight)
    return mm_res.float().mul_(weight_scale * output_scale).to(output.dtype)


class LinearInt8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight, weight_scale)
        return int8_forward_tokenwise(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False, False):
            raise NotImplementedError("Int A8W8 cannot be used for full finetuning")

        weight, weight_scale = ctx.saved_tensors
        return int8_backward_axiswise(output, weight, weight_scale), None, None, None, None

class LinearFp8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight, weight_scale)
        return fp8_forward_tokenwise(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False, False):
            raise NotImplementedError("Float A8W8 cannot be used for full finetuning")

        weight, weight_scale = ctx.saved_tensors
        return fp8_backward_axiswise(output, weight, weight_scale), None, None, None, None

class LinearW8A8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

        self.__is_quantized = False
        self.compute_dtype = None
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return dequantize(self.weight.detach(), self.scale).to(dtype)

    @torch.no_grad()
    def quantize(self, device: torch.device | None = None):
        if self.__is_quantized:
            return
        self.__is_quantized = True

        weight = self.weight.detach()
        orig_device = weight.device
        if device is not None:
            weight = weight.to(device=device)
        if self._dtype == torch.int8:
            weight, scale = quantize_int8_tensorwise(weight)
        else:
            weight, scale = quantize_fp8_tensorwise(weight)

        if device is not None:
            weight = weight.to(device=orig_device)

        self.requires_grad_(False)
        self.weight.data = weight

        self.scale.copy_(scale)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        assert self.__is_quantized
        x = x_orig.reshape(-1, x_orig.shape[-1])

        if x.shape[0] > 16:
            if self._dtype == torch.int8:
                y = LinearInt8Function.apply(x, self.weight, self.scale, self.bias, self.compute_dtype)
            else:
                y = LinearFp8Function.apply(x, self.weight, self.scale, self.bias, self.compute_dtype)
        else:
            w = dequantize(self.weight.detach(), self.scale)
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

def run_benchmark(fn, desc, steps=10000, warmup=500, compile=False):
    if compile:
        fn = torch.compile(fn, fullgraph=True)
    from tqdm import tqdm
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in tqdm(range(steps), desc=desc):
        fn()
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_int8(m, k, n, device = 'cuda'):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.int8)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.int8)
    w_8 = torch.ones (n,k, device=device, dtype=torch.int8)
    w_scale = torch.ones(1, device=device)


    run_benchmark(lambda: torch._int_mm(x_8, w_8.T), "torch mm int")
    run_benchmark(lambda: mm_8bit(x_8, w_8.T), "triton mm int")
    def torch_backward(a, b):
        torch._int_mm(a, b.T.contiguous().T)
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward int8")
    run_benchmark(lambda: mm_8bit(y_8, w_8), "triton mm backward int8")

    run_benchmark(lambda: int8_forward_tokenwise(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "torch forward int", compile=True)
    run_benchmark(lambda: int8_backward_axiswise(y, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "triton backward int", compile=True)


@torch.no_grad()
def benchmark_fp8(m, k, n, device = 'cuda'):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.float8_e4m3fn)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.float8_e4m3fn)
    w_8 = torch.ones (n,k, device=device, dtype=torch.float8_e4m3fn)
    w_scale = torch.ones(1, device=device, dtype=torch.bfloat16)
    one_scale = torch.ones(1, device=device)

    run_benchmark(lambda: torch._scaled_mm(x_8, w_8.T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float()), "torch mm fp8")
    run_benchmark(lambda: mm_8bit(x_8, w_8.T), "triton mm fp8")
    def torch_backward(a, b):
        torch._scaled_mm(a, b.T.contiguous().T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float())
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward fp8")
    run_benchmark(lambda: mm_8bit(y_8, w_8), "triton mm backward fp8")
    run_benchmark(lambda: fp8_forward_tokenwise(x, w_8, w_scale), "torch forward fp8", compile=True)
    run_benchmark(lambda: fp8_backward_axiswise(y, w_8, w_scale), "triton backward fp8", compile=True)


if __name__ == "__main__":
    benchmark_int8(2 * 1024 + 50, 3072, 3072 + 16)
    benchmark_fp8(2 * 1024 + 50, 3072, 3072 + 16)
