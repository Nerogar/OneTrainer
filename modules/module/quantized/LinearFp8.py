from modules.module.quantized.mixin.CompressedWeightMixin import CompressedWeightMixin
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch
from torch import nn


class LinearFp8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
    CompressedWeightMixin,
):
    is_quantized: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_quantized = False

        self.fp8_dtype = torch.float8_e4m3fn
        self._scale = torch.tensor(1.0, dtype=torch.float)
        self.register_buffer("scale", self._scale)
        self.compute_dtype = None

        self._init_compressed_state()

    def original_weight_shape(self) -> tuple[int, ...]:
        if self._compressed:
            return self._weight_shape
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight.detach()
        if self._scale is not None:
            return weight.to(dtype) * self._scale.to(dtype=dtype)
        else:
            return weight.to(dtype=dtype)

    def quantize(self, device: torch.device | None = None):
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if weight.dtype != self.fp8_dtype:
            if device is not None:
                weight = weight.to(device=device)

            abs_max = weight.abs().max()
            self._scale.copy_(torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max)
            weight = weight.div_(self._scale).to(dtype=self.fp8_dtype)

            if device is not None:
                weight = weight.to(device=orig_device)
        self.weight.data = weight

        if self.compress:
            self._compress_weight(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # decode the resident blob to fp8; compute stays bf16 via the plain F.linear below, so
        # backward is unchanged (autograd holds the bf16 weight, exactly as in the uncompressed path)
        weight = self._decompress(self.weight.detach()) if self._compressed else self.weight.detach()
        weight = weight.to(dtype=self.compute_dtype if self.compute_dtype is not None else x.dtype)

        if self._scale is not None:
            weight = weight.mul_(self._scale)
        x = nn.functional.linear(x, weight, self.bias)

        return x
