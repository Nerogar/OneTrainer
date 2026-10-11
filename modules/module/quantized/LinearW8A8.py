from modules.module.quantized.mixin.CompressedWeightMixin import CompressedWeightMixin
from modules.module.quantized.mixin.LoRAFusableLinearMixin import LoRAFusableLinearMixin
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    dequantize,
    quantize_axiswise,
    quantize_fp8_axiswise,
    quantize_fp8_tensorwise,
    quantize_int8_axiswise,
    quantize_int8_tensorwise,
)

import torch
from torch import Tensor, nn


@torch.no_grad()
def forward_tokenwise_postscaled_torch(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    if weight.dtype == torch.int8:
        x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
        res = torch._int_mm(x_8, weight.T)
        res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    else:
        x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
        one = torch.tensor(1.0, device=x.device)
        res = torch._scaled_mm(x_8, weight.T, scale_a=one, scale_b=weight_scale.float(), out_dtype=torch.float)
        res_scaled = res.mul_(x_scale).to(compute_dtype) #much faster than scaled by _scaled_mm
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def forward_tokenwise_epiloguescaled_triton(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_axiswise(x, dim=-1, dtype=weight.dtype)
    res_scaled = mm_8bit(x_8, weight.T, out_dtype=compute_dtype, scale_m=weight_scale * x_scale)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def backward_tokenwise_epiloguescaled_triton(output: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
    output_8, output_scale = quantize_axiswise(output, dim=-1, dtype=weight.dtype)
    #almost always, grad outputs are already contiguous and this is a no-op. But there are some grad outputs from SDXL that are non-contiguous:
    return mm_8bit(output_8.contiguous(), weight, out_dtype=output.dtype, scale_m=weight_scale * output_scale)


forward_tokenwise = forward_tokenwise_epiloguescaled_triton
backward_tokenwise = backward_tokenwise_epiloguescaled_triton


@torch.no_grad()
def forward_tokenwise_lora_epiloguescaled_triton(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype, x_down: Tensor, lora_up: Tensor) -> Tensor:
    x_8, x_scale = quantize_axiswise(x, dim=-1, dtype=weight.dtype)
    res_scaled = mm_8bit(x_8, weight.T, out_dtype=compute_dtype, scale_m=weight_scale * x_scale, lora_xd=x_down, lora_up=lora_up)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def backward_tokenwise_lora_epiloguescaled_triton(output: Tensor, weight: Tensor, weight_scale: Tensor, grad_x_down_pre: Tensor, lora_down: Tensor) -> Tensor:
    output_8, output_scale = quantize_axiswise(output, dim=-1, dtype=weight.dtype)
    return mm_8bit(output_8.contiguous(), weight, out_dtype=output.dtype, scale_m=weight_scale * output_scale, lora_xd=grad_x_down_pre, lora_up=lora_down.to(grad_x_down_pre.dtype))


forward_tokenwise_lora = forward_tokenwise_lora_epiloguescaled_triton
backward_tokenwise_lora = backward_tokenwise_lora_epiloguescaled_triton


class LinearW8A8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        # `weight` is the decompressed weight, so saving it keeps a full-size copy alive until backward.
        # Under reentrant checkpointing that copy lives only inside the recomputed segment, and not saving
        # it would cost a second decode: there is no partitioner, so recompute would decode once for the
        # forward and backward would decode again.
        # TODO once offloading uses non-reentrant checkpointing, consider not saving the decompressed
        # weight and decoding in backward() instead.
        ctx.save_for_backward(weight, weight_scale)
        return forward_tokenwise(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False, False):
            raise NotImplementedError("Int/Float A8W8 cannot be used for full finetuning")

        weight, weight_scale = ctx.saved_tensors
        return backward_tokenwise(output, weight, weight_scale), None, None, None, None


class LinearW8A8LoRAFunction(torch.autograd.Function):
    #LinearW8A8Function plus a low-rank update: it owns the down-projection and the dropout, so the LoRA
    #dgrad folds into the backward epilogue and the (M, out) product is never materialized. The weight is
    #the already decompressed one, as in LinearW8A8Function
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype, lora_down: Tensor, lora_up: Tensor, dropout_mask: Tensor | None) -> Tensor:
        x_down = torch.nn.functional.linear(x, lora_down)
        if dropout_mask is not None:
            x_down = x_down * dropout_mask
        #backward runs outside autocast, so x is saved in x_down's dtype: grad_lora_down is a
        #compute-dtype product there, as it is in the unfused path where autocast casts x
        ctx.save_for_backward(weight, weight_scale, x.to(x_down.dtype), lora_down, lora_up, dropout_mask, x_down)
        return forward_tokenwise_lora(x, weight, weight_scale, bias, compute_dtype, x_down, lora_up)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.needs_input_grad[1:5] != (False, False, False, False):
            raise NotImplementedError("Int/Float A8W8 cannot be used for full finetuning")

        weight, weight_scale, x, lora_down, lora_up, dropout_mask, x_down = ctx.saved_tensors
        needs_x, needs_down, needs_up = ctx.needs_input_grad[0], ctx.needs_input_grad[5], ctx.needs_input_grad[6]
        #grad_x_down_pre (post-dropout-backward) feeds both the grad_x epilogue fold and grad_lora_down
        grad_x_down_pre = None
        if needs_x or needs_down:
            grad_x_down_pre = grad_output @ lora_up.T
            if dropout_mask is not None:
                grad_x_down_pre = grad_x_down_pre * dropout_mask
        grad_x = backward_tokenwise_lora(grad_output, weight, weight_scale, grad_x_down_pre, lora_down) if needs_x else None
        grad_lora_down = (grad_x_down_pre.T @ x).to(lora_down.dtype) if needs_down else None
        grad_lora_up = x_down.T @ grad_output if needs_up else None
        return grad_x, None, None, None, None, grad_lora_down, grad_lora_up, None


class LinearW8A8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
    CompressedWeightMixin,
    LoRAFusableLinearMixin,
):
    def __init__(self, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

        self.__is_quantized = False
        self.compute_dtype = None
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

        self._init_compressed_state()

    def original_weight_shape(self) -> tuple[int, ...]:
        if self._compressed:
            return self._weight_shape
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight.detach()
        # 'scale' is not offloaded, so it can sit on the train device while 'weight' is parked on the temp device
        return dequantize(weight, self.scale.to(device=weight.device)).to(dtype)

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

        if self.compress:
            self._compress_weight(device=device)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        assert self.__is_quantized
        x = x_orig.reshape(-1, x_orig.shape[-1])

        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight

        if x.shape[0] > 16:
            y = LinearW8A8Function.apply(x, weight, self.scale, self.bias, self.compute_dtype)
        else:
            w = dequantize(weight.detach(), self.scale)
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

    def forward_with_lora(self, x_orig: torch.Tensor, lora_down: torch.Tensor, lora_up: torch.Tensor, dropout: torch.nn.Dropout, alpha: torch.Tensor) -> torch.Tensor:
        assert self.__is_quantized
        lora_rank = lora_down.shape[0]

        x = x_orig.reshape(-1, x_orig.shape[-1])
        if x.shape[0] <= 16:
            ld = torch.nn.functional.linear(dropout(torch.nn.functional.linear(x_orig, lora_down)), lora_up)
            return LinearW8A8.forward(self, x_orig) + ld * (alpha / lora_rank)

        #the cast matches lora_up (kept in its own dtype, e.g. f32) to the down-projection's autocast
        #dtype, which the epilogue dot needs; dropout(ones) yields the scaled 0/(1/(1-p)) mask, drawn
        #here rather than inside LinearW8A8LoRAFunction so the RNG stays inductor-native
        lora_up_scaled = (lora_up * (alpha / lora_rank)).T.to(self.compute_dtype)
        dropout_mask = dropout(torch.ones(x.shape[0], lora_rank, device=x.device, dtype=self.compute_dtype)) if (dropout.training and dropout.p > 0) else None
        y = self._fused_lora_forward(x, lora_down, lora_up_scaled, dropout_mask)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

    #x is 2D, lora_down is (r, in_features), lora_up is (r, out_features) with alpha already folded in
    def _fused_lora_forward(self, x: Tensor, lora_down: Tensor, lora_up: Tensor, dropout_mask: Tensor | None) -> Tensor:
        assert not self.weight.requires_grad
        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight
        return LinearW8A8LoRAFunction.apply(x, weight, self.scale, self.bias, self.compute_dtype, lora_down, lora_up, dropout_mask)

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
def benchmark_int8(m, k, n, device = 'cuda', steps = 10000):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.int8)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.int8)
    w_8 = torch.ones (n,k, device=device, dtype=torch.int8)
    w_scale = torch.ones(1, device=device)


    run_benchmark(lambda: torch._int_mm(x_8, w_8.T), "torch mm int", steps=steps)
    run_benchmark(lambda: mm_8bit(x_8, w_8.T, out_dtype=torch.int32), "triton mm int", steps=steps)
    def torch_backward(a, b):
        torch._int_mm(a, b.T.contiguous().T)
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward int8", steps=steps)
    run_benchmark(lambda: mm_8bit(y_8, w_8, out_dtype=torch.int32), "triton mm backward int8", steps=steps)

    run_benchmark(lambda: forward_tokenwise_postscaled_torch(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "torch forward int", steps=steps, compile=True)
    run_benchmark(lambda: forward_tokenwise_epiloguescaled_triton(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "triton scaled forward int", steps=steps, compile=True)
    run_benchmark(lambda: backward_tokenwise_epiloguescaled_triton(y, w_8, w_scale), "triton scaled backward int", steps=steps, compile=True)


@torch.no_grad()
def benchmark_fp8(m, k, n, device = 'cuda', steps = 10000):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.float8_e4m3fn)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.float8_e4m3fn)
    w_8 = torch.ones (n,k, device=device, dtype=torch.float8_e4m3fn)
    w_scale = torch.ones(1, device=device, dtype=torch.bfloat16)
    one_scale = torch.ones(1, device=device)

    run_benchmark(lambda: torch._scaled_mm(x_8, w_8.T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float()), "torch mm fp8", steps=steps)
    run_benchmark(lambda: mm_8bit(x_8, w_8.T, out_dtype=torch.float32), "triton mm fp8", steps=steps)
    def torch_backward(a, b):
        torch._scaled_mm(a, b.T.contiguous().T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float())
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward fp8", steps=steps)
    run_benchmark(lambda: mm_8bit(y_8, w_8, out_dtype=torch.float32), "triton mm backward fp8", steps=steps)

    run_benchmark(lambda: forward_tokenwise_postscaled_torch(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "torch forward fp8", steps=steps, compile=True)
    run_benchmark(lambda: forward_tokenwise_epiloguescaled_triton(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "triton scaled forward fp8", steps=steps, compile=True)
    run_benchmark(lambda: backward_tokenwise_epiloguescaled_triton(y, w_8, w_scale), "triton scaled backward fp8", steps=steps, compile=True)


@torch.no_grad()
def benchmark_lora(m, k, n, r, dtype, device='cuda', steps=1000):
    is_int8 = (dtype == torch.int8)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    if is_int8:
        w_8 = torch.randint(-127, 127, (n, k), device=device, dtype=torch.int8)
    else:
        w_8 = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
    w_scale = torch.full((1,), 0.01, device=device)
    down_w = torch.randn(r, k, device=device, dtype=torch.bfloat16) * 0.02
    up_w = torch.randn(n, r, device=device, dtype=torch.bfloat16) * 0.02

    baseline = forward_tokenwise
    fused = forward_tokenwise_lora_epiloguescaled_triton

    def run_unfused():
        y = baseline(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16)
        return y + (x @ down_w.T) @ up_w.T

    def run_fused():
        x_down = x @ down_w.T
        return fused(x, w_8, w_scale, None, torch.bfloat16, x_down, up_w.T)

    name = "int8" if is_int8 else "fp8"
    diff = (run_unfused().float() - run_fused().float()).abs().max().item()
    ref = run_unfused().float().abs().mean().item()
    print(f"lora {name} m={m} k={k} n={n} r={r}: max abs diff fused vs unfused = {diff:.4f} (mean magnitude {ref:.3f})")
    run_benchmark(lambda: baseline(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), f"no-lora {name} baseline", steps=steps, compile=True)
    run_benchmark(run_unfused, f"unfused lora {name}", steps=steps, compile=True)
    run_benchmark(run_fused, f"fused lora {name}", steps=steps, compile=True)


@torch.no_grad()
def benchmark_lora_backward(m, k, n, r, dtype, device='cuda', steps=1000):
    #isolates the grad_x path that the backward fusion changes: unfused = main bwd mm +
    #standalone (M,K) lora dgrad + merge add; fused = the same lora dgrad folded into the
    #bwd mm epilogue. grad_x_down (= grad_y @ up.T) is shared and timed in both; the
    #grad_lora_up / grad_lora_down gemms are identical in both variants and excluded.
    is_int8 = (dtype == torch.int8)
    grad_y = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    if is_int8:
        w_8 = torch.randint(-127, 127, (n, k), device=device, dtype=torch.int8)
    else:
        w_8 = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
    w_scale = torch.full((1,), 0.01, device=device)
    down_w = torch.randn(r, k, device=device, dtype=torch.bfloat16) * 0.02
    up_w = torch.randn(n, r, device=device, dtype=torch.bfloat16) * 0.02

    main_bwd = backward_tokenwise
    fused_bwd = backward_tokenwise_lora

    def run_unfused():
        grad_x = main_bwd(grad_y, w_8, w_scale)
        grad_x_down = grad_y @ up_w
        return grad_x + grad_x_down @ down_w

    def run_fused():
        grad_x_down = grad_y @ up_w
        return fused_bwd(grad_y, w_8, w_scale, grad_x_down, down_w)

    name = "int8" if is_int8 else "fp8"
    diff = (run_unfused().float() - run_fused().float()).abs().max().item()
    ref = run_unfused().float().abs().mean().item()
    print(f"lora-bwd {name} m={m} k={k} n={n} r={r}: max abs diff fused vs unfused = {diff:.4f} (mean magnitude {ref:.3f})")
    run_benchmark(run_unfused, f"unfused lora-bwd {name}", steps=steps, compile=True)
    run_benchmark(run_fused, f"fused lora-bwd {name}", steps=steps, compile=True)


if __name__ == "__main__":
    #ragged shape: M and the backward's contraction over 3088 are both non-%128, so the
    #masked-loop fallback and, with r=12, the padded rank tile are covered
    benchmark_int8(2 * 1024 + 50, 3072, 3072 + 16)
    benchmark_fp8(2 * 1024 + 50, 3072, 3072 + 16)
    benchmark_lora(2 * 1024 + 50, 3072, 3072 + 16, r=12, dtype=torch.int8)
    benchmark_lora_backward(2 * 1024 + 50, 3072, 3072 + 16, r=12, dtype=torch.int8)
    #a real FLUX.2-klein-9B shape at 512px batch 2: M = 2*(1024 image + 512 text) tokens
    #through a double block's 4096 -> 4096 attention projection
    benchmark_lora(2 * (1024 + 512), 4096, 4096, r=16, dtype=torch.int8, steps=1000)
