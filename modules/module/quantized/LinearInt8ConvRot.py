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

# Tuning knobs, env-driven rather than config-plumbed: these select between behaviours that are
# still being characterized, so they are deliberately not exposed in the UI.
# Block size for the group-wise Hadamard rotation. Not empirically validated; 128 is a common
# group size in block quantization literature, picked as a starting point to sweep from.
CONVROT_BLOCK_SIZE = int(os.environ.get("CONVROT_BLOCK_SIZE", "128"))
# Isolates the part of the design that is still unmeasured: whether quantizing the backward
# grad-output to int8 (matching plain int8w8's backward) hurts LoRA gradient quality vs keeping
# it in bf16. Default matches int8w8 so the two paths are directly comparable.
CONVROT_BF16_DY = os.environ.get("CONVROT_BF16_DY", "0") == "1"
# Which forward to run once the weight is stored as rotated int8. "int8" uses the fused kernel; "bf16"
# dequantizes the weight and hands the matmul to the vendor BF16 GEMM. Accuracy and speed pull in opposite
# directions, so this is a trade rather than a free choice.
#
# Measured here on a 5090, 5376 -> 7168, against the dequantized weight as reference:
#
#     tokens    fused int8    bf16
#      2048       1.27 ms    2.74 ms
#     12000      10.13 ms   11.70 ms
#     error      9.43e-03   3.61e-03
#
# So bf16 is ~2.6x more accurate and 1.15-2.2x slower. Both routes carry the rotation's own quantization error
# identically -- that is fixed when the weight is stored -- and what separates them is that the fused kernel
# additionally quantizes the ACTIVATIONS per row on every call.
#
# musubi measures bf16 as both more accurate AND faster on an H100 (4.10 vs 6.22 s/it), noting "INT8 arithmetic
# is not the problem, since an H100 runs INT8 and FP8 at the same rate -- the fused kernel is". That half does
# not transfer: torch._int_mm on consumer hardware is fast, so the speed ordering reverses while the accuracy
# ordering holds. Benchmark before assuming either.
#
# Default int8, which is the faster route here and matches the behaviour before this knob existed. Prefer bf16
# when gradient quality matters more than throughput: the base is frozen, so a LoRA's entire gradient arrives
# through this layer, and activation quantization lands directly on it.
CONVROT_FWD = os.environ.get("CONVROT_FWD", "int8").lower()


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
        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight.detach()
        w = dequantize(weight, self.scale.to(device=weight.device))
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
        # assign through .data rather than rebinding self.weight, so the Parameter object's identity
        # survives quantization -- matches every other quantized layer, and anything already holding
        # a reference to it (offloading, param groups) keeps seeing the live tensor
        self.weight.data = weight
        self.scale.copy_(scale)

        # this override replaces LinearW8A8.quantize entirely, so the compression step has to be
        # repeated here; forward() decompresses on the way in
        if self.compress:
            self._compress_weight(device=device)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        assert self._is_quantized
        x = x_orig.reshape(-1, x_orig.shape[-1])

        # The row floor is a HARD REQUIREMENT of torch._int_mm, not a performance heuristic -- do not remove it
        # or relax it to >=. Measured on torch 2.12.0+cu130 / RTX 5090, int8 5376 -> 7168:
        #     rows 15 -> RuntimeError: self.size(0) needs to be greater than 16, but got 15
        #     rows 16 -> RuntimeError: self.size(0) needs to be greater than 16, but got 16
        #     rows 17 -> ok
        # so 16 itself fails and strictly-greater is correct. Anything at or below it must take the bf16 route,
        # which is why that branch is the fallback rather than an alternative: it is the only path that works
        # for small inputs, whatever CONVROT_FWD asks for.
        if CONVROT_FWD == "int8" and x.shape[0] > 16:
            # Decompress first: LinearW8A8 inherits CompressedWeightMixin, so self.weight may be a compressed
            # blob rather than the int8 tensor, and LinearInt8ConvRotFunction consumes the int8 weight directly.
            weight = self._decompress(self.weight) if self._compressed else self.weight
            y = LinearInt8ConvRotFunction.apply(
                x, weight, self.scale, self.bias, self.compute_dtype,
                self.block_size, self.in_features, self.rotation,
            )
        else:
            # Rotate the ACTIVATIONS and leave the weight rotated as stored, rather than un-rotating the
            # weight. The Hadamard is orthogonal, so (xR)(WR)^T == x W^T either way, but the costs are not
            # symmetric: rotating x is tokens x in_features, un-rotating W is out_features x in_features and
            # runs on every call. Measured on 12000x5376 -> 7168, this is the whole difference between the
            # bf16 route being slower than the fused int8 kernel and being faster.
            #
            # unquantized_weight() still un-rotates, because its callers -- LoRA decompose, offload sizing --
            # want the weight in its original basis. Only this forward can exploit the stored basis directly.
            weight = self._decompress(self.weight) if self._compressed else self.weight
            w_rotated = dequantize(weight, self.scale.to(device=weight.device)).to(x.dtype)
            x_rotated = block_hadamard(pad_to_block(x, self.block_size), self.block_size, self.rotation)
            y = torch.nn.functional.linear(x_rotated, w_rotated, self.bias)

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
    # Clock/config caveats: throughput numbers here are indicative only - they are taken without
    # locked clocks, so treat them as relative comparisons between the routes rather than absolute
    # figures.
    benchmark_snr()
    print()
    benchmark_throughput(2 * 1024 + 50, 3072, 3088)
