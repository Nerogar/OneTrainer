from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch
from torch import Tensor, nn


def _right_multiply_blocks(x: Tensor, rot: Tensor, transpose: bool) -> Tensor:
    # Right-multiplies the last dim by the block-diagonal orthogonal matrix in `rot`
    # ((r, b, b) -> blocks of b along the last dim): x @ rot, or x @ rot^T when transpose.
    shape = x.shape
    r, b = rot.shape[0], rot.shape[1]
    rt = rot.transpose(-1, -2) if transpose else rot
    x3 = x.reshape(-1, r, b).permute(1, 0, 2)  # (r, rows, b)
    out = torch.bmm(x3, rt.to(x3.dtype))
    return out.permute(1, 0, 2).reshape(*shape[:-1], -1)


# Fixed base for the per-block mix seeds: the rotation must be identical in every process
# that quantizes the same weight, so the random part is a seeded RNG, never torch.randn.
_ROT_SEED = 0x5EED


def _random_orthogonal(size: int, seed: int, device: torch.device) -> Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(size, size, generator=g)
    u, _, vh = torch.linalg.svd(a.to(device), full_matrices=False)
    return u @ vh


@torch.no_grad()
def _build_rotation(w: Tensor, block_size: int) -> Tensor:
    # Per-block rotation O_k = V_k^T M_k, the block-granular analogue of SVDQuant's
    # rotate-then-quantize. SVD of each (n x b) input-channel block W_k = U_k S_k V_k^T gives
    # the data-dependent V_k; M_k is a seeded random orthogonal mix. It is the mix that
    # spreads outlier energy across the block's channels before the quantizer sees them (an
    # SVD-alignment alone would concentrate the outlier in the top singular direction); the V
    # alignment keeps the rotation data-dependent. The stored weight becomes W_k O_k^T and the
    # module applies O_k^T to its inputs, so it still computes W_k x. Deterministic: the same
    # weight always derives the same rotation.
    n, k = w.shape
    r = k // block_size
    w3 = w.reshape(n, r, block_size).permute(1, 0, 2)  # (r, n, b): one n x b matrix per block
    _, _, vh = torch.linalg.svd(w3, full_matrices=False)  # (r, b, b): V_k^T
    o = vh.clone()
    for i in range(r):
        o[i] = o[i] @ _random_orthogonal(block_size, _ROT_SEED + i, w.device)
    return o  # (r, b, b): O_k = V_k^T M_k


class BaseLinearQuantRot(
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    pass


def make_quant_rot_linear(linear_class):
    # Orthogonal-SVDQuant: the base weight is quantized as W @ O with O a block-diagonal orthogonal
    # matrix derived from W's own SVD (outliers spread evenly across each block), and the module
    # applies O^T to its inputs in forward, so the module still computes W x.
    class LinearQuantRot(
        linear_class,
        BaseLinearQuantRot,
    ):
        def __init__(self, quant_rot_block_size: int, quant_rot_dtype: torch.dtype, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if quant_rot_block_size <= 0:
                raise ValueError(f"quant_rot_block_size must be positive, got {quant_rot_block_size}")
            self.__quant_rot_is_quantized = False
            self.__quant_rot_block_size = quant_rot_block_size
            self.__quant_rot_dtype = quant_rot_dtype
            in_features = kwargs.get('in_features')
            self.__rotatable = in_features is None or in_features % quant_rot_block_size == 0
            self.quant_rot = torch.nn.Parameter(torch.empty(0), requires_grad=False)

        @property
        def quant_rot_block_size(self) -> int:
            return self.__quant_rot_block_size

        @property
        def quant_rot_dtype(self) -> torch.dtype:
            return self.__quant_rot_dtype

        @property
        def is_rotatable(self) -> bool:
            return self.__rotatable and self.quant_rot.numel() > 0

        def _rotatable(self) -> bool:
            return self.is_rotatable

        @torch.no_grad()
        def quantize(self, device: torch.device | None = None):
            if self.__quant_rot_is_quantized:
                return

            if self.__rotatable:
                W = super().unquantized_weight(torch.float32, self.weight.device)
                rot = _build_rotation(W, self.__quant_rot_block_size).to(self.__quant_rot_dtype)
                self.quant_rot = torch.nn.Parameter(rot, requires_grad=False)
                # quantize the rotated weight W @ O: the orthogonal mix spreads each block's
                # outliers across all of its channels before the quantizer sees them.
                rotated = _right_multiply_blocks(W, rot, transpose=True).to(
                    dtype=self.weight.dtype, device=self.weight.device)
                self.weight.data = rotated
                del W, rot

            # set the flag only once the rotation work has succeeded: on a failure the module
            # would otherwise be left rotated (weight.data = W @ O^T) with an empty quant_rot,
            # and forward would silently apply the weight without O^T.
            self.__quant_rot_is_quantized = True
            super().quantize(device=device)

        def forward(self, x: Tensor) -> Tensor:
            assert self.__quant_rot_is_quantized
            if self._rotatable():
                # x @ O^T per block, so that (W @ O^T)(x @ O^T) = W x up to quantization error.
                x = _right_multiply_blocks(x, self.quant_rot, transpose=True)
            return super().forward(x)

        def forward_rotated_input(self, x: Tensor) -> Tensor:
            # For a caller that already folded O^T into the rotation it applied to x (the
            # OFTModule merged path): use the stored (W @ O^T) weight as-is, skipping O^T.
            assert self.__quant_rot_is_quantized
            return super().forward(x)

        def forward_with_lora(self, x: Tensor, lora_down: nn.Linear, lora_up: nn.Linear, dropout: nn.Dropout, alpha: float) -> Tensor:
            # out = base(x) + lora_up(lora_down(x)) * (alpha / rank), where the LoRA delta acts in
            # the ORIGINAL input space (x, not x @ O^T), so the adapter learns W + lora_up lora_down^T
            # exactly like the unquantized LoRA case. base(x) is reproduced here (instead of
            # self.forward) because self.forward is replaced by the adapter hook once applied.
            assert self.__quant_rot_is_quantized
            assert lora_down.bias is None and lora_up.bias is None
            x_base = _right_multiply_blocks(x, self.quant_rot, transpose=True) if self._rotatable() else x
            # super() dispatches to the quantized base (LinearW8A8 / LinearSVD / LinearNf4) forward,
            # which applies the stored (rotated, quantized) weight; with SVDQuant it also adds the SVD
            # low-rank part, which acts on the same rotated input x_base (A @ (x @ O^T)).
            base = super().forward(x_base)
            ld = lora_up(dropout(lora_down(x)))
            return base + ld * (alpha / lora_down.weight.shape[0])

        def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
            if self.__quant_rot_is_quantized and self._rotatable():
                # undo the rotation: W = (W @ O^T) @ O
                w = super().unquantized_weight(dtype, device)
                return _right_multiply_blocks(w, self.quant_rot.to(device=w.device), transpose=False)
            return super().unquantized_weight(dtype, device)

    return LinearQuantRot
