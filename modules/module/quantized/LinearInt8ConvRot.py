import os

from modules.module.quantized.LinearW8A8 import (
    LinearW8A8,
    int8_backward_axiswise,
    int8_forward_tokenwise,
    run_benchmark,
)
from modules.util.hadamard import block_hadamard, hadamard_matrix, pad_to_block
from modules.util.quantization_util import (
    dequantize,
    quantize_int8_tensorwise,
)

import torch
from torch import Tensor

# Prototype knobs - env-driven, no UI/config plumbing (see PLAN in-an-new-branch-reflective-cocke).
# Block size for the group-wise Hadamard rotation. Not empirically validated; 128 is a common
# group size in block quantization literature, picked as a starting point to sweep from.
CONVROT_BLOCK_SIZE = int(os.environ.get("CONVROT_BLOCK_SIZE", "128"))
# Isolates the untested part of this experiment: whether quantizing the backward grad-output
# to int8 (matching plain int8w8's backward) hurts LoRA gradient quality vs keeping it in
# bf16. Default matches int8w8 so the two paths are directly comparable.
CONVROT_BF16_DY = os.environ.get("CONVROT_BF16_DY", "0") == "1"


class LinearInt8ConvRotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None,
                compute_dtype: torch.dtype, block_size: int, in_features: int, rotation: Tensor) -> Tensor:
        ctx.save_for_backward(weight, weight_scale, rotation)
        ctx.block_size = block_size
        ctx.in_features = in_features

        x_rotated = block_hadamard(pad_to_block(x, block_size), block_size, rotation)
        return int8_forward_tokenwise(x_rotated, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad != (True, False, False, False, False, False, False, False):
            raise NotImplementedError("Int A8W8 ConvRot cannot be used for full finetuning")

        weight, weight_scale, rotation = ctx.saved_tensors
        block_size, in_features = ctx.block_size, ctx.in_features

        if CONVROT_BF16_DY:
            # Debug path: skip int8 quantization of the grad-output entirely, to measure how
            # much of any gradient-quality gap comes from quantizing dY vs. the rotation itself.
            w_true = block_hadamard(dequantize(weight, weight_scale), block_size, rotation)[..., :in_features]
            dx = output.to(torch.float32) @ w_true
        else:
            dx_padded = int8_backward_axiswise(output, weight, weight_scale)
            dx = block_hadamard(dx_padded, block_size, rotation)[..., :in_features]

        return dx.to(output.dtype), None, None, None, None, None, None, None


class LinearInt8ConvRot(LinearW8A8):
    # Group-wise Hadamard-rotated INT8 W8A8: rotates activations/weight/grad-output along the
    # contraction dim in independent blocks of `block_size` before int8 quantization, so
    # channel outliers get spread across a block instead of dominating a single channel's
    # quantization range. The GEMM itself is unchanged from LinearW8A8 (still torch._int_mm) -
    # rotation is the only added op, applied via block_hadamard (see modules/util/hadamard.py).
    def __init__(self, block_size: int | None = None, *args, **kwargs):
        kwargs['dtype'] = torch.int8  # ConvRot is int8-only; no fp8 variant in this experiment.
        super().__init__(*args, **kwargs)
        self.block_size = block_size if block_size is not None else CONVROT_BLOCK_SIZE
        self._is_quantized = False  # own flag; LinearW8A8's is name-mangled and not reachable here

    def original_weight_shape(self) -> tuple[int, ...]:
        return (self.out_features, self.in_features)

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = dequantize(self.weight.detach(), self.scale)
        w = block_hadamard(w, self.block_size, self.rotation)[..., :self.in_features]
        return w.to(dtype=dtype, device=device)

    @torch.no_grad()
    def quantize(self, device: torch.device | None = None):
        if self._is_quantized:
            return
        self._is_quantized = True

        weight = self.weight.detach()
        orig_device = weight.device
        if device is not None:
            weight = weight.to(device=device)

        # Precompute the block-Hadamard matrix once and keep it as a buffer (fp32; block_hadamard
        # casts to the operand dtype). forward/backward run inside torch.compile'd + checkpointed
        # blocks where the lazy hadamard_matrix() cache write is an illegal in-graph side effect;
        # a buffer travels with the module across devices and is only read in the graph. persistent
        # is False so it stays out of the state_dict (it's derivable from block_size).
        self.register_buffer("rotation", hadamard_matrix(self.block_size, orig_device, torch.float32), persistent=False)

        rotated = block_hadamard(pad_to_block(weight, self.block_size), self.block_size, self.rotation.to(weight.device))
        weight, scale = quantize_int8_tensorwise(rotated)

        if device is not None:
            weight = weight.to(device=orig_device)

        self.requires_grad_(False)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale.copy_(scale)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        assert self._is_quantized
        x = x_orig.reshape(-1, x_orig.shape[-1])

        if x.shape[0] > 16:
            y = LinearInt8ConvRotFunction.apply(
                x, self.weight, self.scale, self.bias, self.compute_dtype,
                self.block_size, self.in_features, self.rotation,
            )
        else:
            w = self.unquantized_weight(dtype=x.dtype, device=x.device)
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1],))


@torch.no_grad()
def quant_relative_error(x: Tensor, block_size: int | None) -> float:
    # Round-trips x through int8 quantization (plain per-tensor if block_size is None, else
    # rotated) and reports ||dequant(quant(x)) - x|| / ||x||: a quick SNR proxy for how much a
    # given block size actually helps, without needing a full training run.
    if block_size is None:
        q, scale = quantize_int8_tensorwise(x)
        recovered = dequantize(q, scale)
    else:
        rotated = block_hadamard(pad_to_block(x, block_size), block_size)
        q, scale = quantize_int8_tensorwise(rotated)
        recovered = block_hadamard(dequantize(q, scale), block_size)[..., :x.shape[-1]]
    return (recovered - x).norm().item() / x.norm().item()


def benchmark_snr(n_channels=3072, n_rows=512, outlier_channels=8, outlier_scale=20.0, device='cuda'):
    # Synthetic outlier-channel weight: a handful of channels scaled up, as in the outlier
    # patterns published PTQ work targets. Plain per-tensor int8 wastes its dynamic range on
    # those few channels; a block Hadamard rotation should spread each outlier's magnitude
    # across its block instead of one channel dominating the whole tensor's scale.
    torch.manual_seed(0)
    w = torch.randn(n_rows, n_channels, device=device)
    outlier_idx = torch.randperm(n_channels)[:outlier_channels]
    w[:, outlier_idx] *= outlier_scale

    print(f"synthetic outlier tensor: {n_channels} channels, {outlier_channels} scaled {outlier_scale}x")
    print(f"  plain int8 (no rotation):        rel err = {quant_relative_error(w, None):.4f}")
    for block_size in (32, 64, 128, 256):
        print(f"  ConvRot block_size={block_size:<4}:            rel err = {quant_relative_error(w, block_size):.4f}")


@torch.no_grad()
def benchmark_throughput(m, k, n, block_size=128, device='cuda'):
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    y = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    w_plain = torch.ones(n, k, device=device, dtype=torch.int8)
    w_scale = torch.ones(1, device=device)

    k_padded = k + (-k) % block_size
    w_rot = torch.ones(n, k_padded, device=device, dtype=torch.int8)

    def convrot_forward():
        xr = block_hadamard(pad_to_block(x, block_size), block_size)
        return int8_forward_tokenwise(xr, w_rot, w_scale, bias=None, compute_dtype=torch.bfloat16)

    def convrot_backward():
        dx = int8_backward_axiswise(y, w_rot, w_scale)
        return block_hadamard(dx, block_size)[..., :k]

    run_benchmark(lambda: int8_forward_tokenwise(x, w_plain, w_scale, bias=None, compute_dtype=torch.bfloat16), "plain int8w8 forward")
    run_benchmark(convrot_forward, "ConvRot forward (incl. rotation)")
    run_benchmark(lambda: int8_backward_axiswise(y, w_plain, w_scale), "plain int8w8 backward")
    run_benchmark(convrot_backward, "ConvRot backward (incl. rotation)")


if __name__ == "__main__":
    # Clock/config caveats: throughput numbers here are indicative only - see the plan's
    # verification section for the locked-clock benchmark this repo normally requires before
    # trusting a number.
    benchmark_snr()
    print()
    benchmark_throughput(2 * 1024 + 50, 3072, 3088)
