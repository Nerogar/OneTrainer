
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.triton_mm_8bit import mm_8bit as triton_mm_8bit

import torch
from torch import Tensor, nn


def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q

def quantize_int8_tensorwise_get_scale(x: Tensor) -> float:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return scale

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    scale = quantize_int8_tensorwise_get_scale(x)
    q = quantize_int8(x, scale)
    return q, scale

def quantize_int8_axiswise_get_scale(x: Tensor, dim: int) -> Tensor:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    scale = quantize_int8_axiswise_get_scale(x, dim)
    q = quantize_int8(x, scale)
    return q, scale

def quantize_fp8(x: Tensor, scale: float | Tensor) -> Tensor:
    q = x.float().mul(1.0 / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q

def quantize_fp8_tensorwise_get_scale(x: Tensor) -> float:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 448.0).clamp(min=1e-30)
    return scale

def quantize_fp8_axiswise_get_scale(x: Tensor, dim: int) -> Tensor:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 448.0).clamp(min=1e-30)
    return scale

def quantize_fp8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    scale = quantize_fp8_tensorwise_get_scale(x)
    q = quantize_fp8(x, scale)
    return q, scale

def quantize_fp8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    scale = quantize_fp8_axiswise_get_scale(x, dim)
    q = quantize_fp8(x, scale)
    return q, scale

def unquantize(q: Tensor, scale: float | Tensor, compute_dtype: torch.dtype) -> Tensor:
    return q.to(compute_dtype) * scale.to(compute_dtype)

def int8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor=None) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.to(x.dtype).mul_(weight_scale * x_scale)
    if bias is not None:
        res_scaled.add_(bias.to(x.dtype))
    return res_scaled

def fp8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor=None) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    one = torch.ones(1, device=x.device)
    res = torch._scaled_mm(x_8, weight.T, scale_a=one, scale_b=weight_scale.float(), out_dtype=x.dtype)
    res_scaled = res.mul_(x_scale) #much faster than scaled by _scaled_mm

    if bias is not None:
        res_scaled.add_(bias.to(x.dtype))
    return res_scaled


def int8_backward_axiswise(x: Tensor, weight: Tensor, weight_scale: float) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    mm_res = triton_mm_8bit(x_8, weight)
    return mm_res.to(x.dtype).mul_(weight_scale * x_scale)

def fp8_backward_axiswise(x: Tensor, weight: Tensor, weight_scale: float) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    mm_res = triton_mm_8bit(x_8, weight)
    return mm_res.to(x.dtype).mul_(weight_scale * x_scale)



class LinearInt8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight, weight_scale)
        return int8_forward_tokenwise(x, weight, weight_scale, bias)

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("Int A8W8 cannot be used for full finetuning")

        weight, weight_scale = ctx.saved_tensors
        return int8_backward_axiswise(x, weight, weight_scale), None, None, None

class LinearFp8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: float, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(weight, weight_scale)
        return fp8_forward_tokenwise(x.bfloat16(), weight, weight_scale, bias).bfloat16()

    @staticmethod
    def backward(ctx, x: Tensor):
        if ctx.needs_input_grad != (True, False, False, False):
            raise NotImplementedError("Float A8W8 cannot be used for full finetuning")

        weight, weight_scale = ctx.saved_tensors
        return fp8_backward_axiswise(x, weight, weight_scale), None, None, None

class LinearW8A8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    def __init__(self, dtype, compute_dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype
        self._compute_dtype = compute_dtype

        self.__is_quantized = False
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return unquantize(self.weight.detach(), self.scale, self._compute_dtype).to(dtype)

    @torch.no_grad()
    def quantize(self, device: torch.device | None = None, **kwargs):
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
        x = x_orig.to(self._compute_dtype).reshape(-1, x_orig.shape[-1])

        if x.shape[0] > 16:
            if self._dtype == torch.int8:
                y = LinearInt8Function.apply(x, self.weight, self.scale, self.bias)
            else:
                y = LinearFp8Function.apply(x, self.weight, self.scale, self.bias)
        else:
            w = unquantize(self.weight, self.scale, compute_dtype=self._compute_dtype)
            y = torch.nn.functional.linear(x, w, self.bias.to(self._compute_dtype))

        assert y.dtype == self._compute_dtype
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
    run_benchmark(lambda: triton_mm_8bit(x_8, w_8.T), "triton mm int")
    def torch_backward(a, b):
        torch._int_mm(a, b.T.contiguous().T)
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward int8")
    run_benchmark(lambda: triton_mm_8bit(y_8, w_8), "triton mm backward int8")

    run_benchmark(lambda: int8_forward_tokenwise(x, w_8, w_scale), "torch forward int", compile=True)
    run_benchmark(lambda: int8_backward_axiswise(y, w_8, w_scale), "triton backward int", compile=True)


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
    run_benchmark(lambda: triton_mm_8bit(x_8, w_8.T), "triton mm fp8")
    def torch_backward(a, b):
        torch._scaled_mm(a, b.T.contiguous().T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float())
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward fp8")
    run_benchmark(lambda: triton_mm_8bit(y_8, w_8), "triton mm backward fp8")
    run_benchmark(lambda: fp8_forward_tokenwise(x, w_8, w_scale), "torch forward fp8", compile=True)
    run_benchmark(lambda: fp8_backward_axiswise(y, w_8, w_scale), "triton backward fp8", compile=True)


if __name__ == "__main__":
    benchmark_int8(2 * 1024 + 50, 3072, 3072 + 16)
    benchmark_fp8(2 * 1024 + 50, 3072, 3072 + 16)
