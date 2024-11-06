from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch
from torch import nn


class LinearFp8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    is_quantized: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_quantized = False

        self.fp8_dtype = torch.float8_e4m3fn
        self.scale = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def quantize(self, device: torch.device | None = None):
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if device is not None:
            weight = weight.to(device=device)

        if weight.dtype != self.fp8_dtype:
            abs_max = weight.abs().max()
            self.scale = torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max
            weight = weight.div_(self.scale).to(dtype=self.fp8_dtype)

        if device is not None:
            weight = weight.to(device=orig_device)
        self.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.detach().to(x.dtype)
        if self.scale is not None:
            weight = weight.mul_(self.scale)
        x = nn.functional.linear(x, weight, self.bias)

        return x
