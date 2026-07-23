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
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float))
        self.compute_dtype = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def mark_needs_requantization(self):
        self.is_quantized = False

    def predict_offload_bytes(self) -> int:
        # weight quantizes to float8_e4m3fn (1 byte/elem, same shape); bias is left unchanged. Matches
        # get_offload_tensors (weight + optional bias); the scalar scale buffer is not offload-counted.
        weight_bytes = self.weight.numel()
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return weight_bytes + bias_bytes

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.scale is not None:
            return self.weight.detach().to(dtype) * self.scale.to(dtype=dtype)
        else:
            return self.weight.detach().to(dtype=dtype)

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
            scale = torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max
            weight = weight.div_(scale).to(dtype=self.fp8_dtype)

            if device is not None:
                weight = weight.to(device=orig_device)

            # keep the scale on the weight's device (see LinearW8A8.quantize)
            self.scale = scale.detach().to(orig_device)
        self.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.detach()
        weight = weight.to(dtype=self.compute_dtype if self.compute_dtype is not None else x.dtype)

        if self.scale is not None:
            weight = weight.mul_(self.scale)
        x = nn.functional.linear(x, weight, self.bias)

        return x
