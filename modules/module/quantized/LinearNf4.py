from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch
from torch import nn

import bitsandbytes as bnb


class LinearNf4(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    is_quantized: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_quantized = False

        self.block_size = 64
        self.nested_block_size = 256

        self._absmax = torch.empty(size=())
        self._offset = torch.empty(size=())
        self._code = torch.empty(size=())
        self._nested_absmax = torch.empty(size=())
        self._nested_code = torch.empty(size=())
        self.shape = self.weight.shape

        self.register_buffer("absmax", self._absmax)
        self.register_buffer("offset", self._offset)
        self.register_buffer("code", self._code)
        self.register_buffer("nested_absmax", self._nested_absmax)
        self.register_buffer("nested_code", self._nested_code)

        self.compute_dtype = None
        self.quant_state = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(self,  dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.is_quantized:
            device_weight = self.weight.to(device=device)
            device_absmax = self._absmax.to(device=device)

            return bnb.functional.dequantize_4bit(
                A=device_weight,
                quant_state=bnb.functional.QuantState(
                    absmax=device_absmax,
                    shape=self.shape,
                    code=self._code,
                    blocksize=self.block_size,
                    quant_type='nf4',
                    dtype=self.compute_dtype,
                    offset=self._offset,
                    state2=self.quant_state.state2,
                ),
                quant_type='nf4',
            ).detach().to(dtype=dtype)
        else:
            return self.weight.detach().to(dtype=dtype)

    def quantize(self, device: torch.device | None = None):
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if weight.dtype != torch.int8:
            if device is not None:
                weight = weight.to(device=device)

            weight, quant_state = bnb.functional.quantize_4bit(
                weight,
                blocksize=self.block_size,
                compress_statistics=True,
                quant_type='nf4',
                quant_storage=torch.uint8,
            )

            self._absmax.data = quant_state.absmax
            self._offset.data = quant_state.offset
            self._code.data = quant_state.code
            self._nested_absmax.data = quant_state.state2.absmax
            self._nested_code.data = quant_state.state2.code

            if device is not None:
                weight = weight.to(device=orig_device)

        # Nf4 weights can not be trained, disable grads for int8 storage
        self.requires_grad_(False)
        self.weight.data = weight

        self.quant_state = bnb.functional.QuantState(
            absmax=self._absmax,
            shape=self.shape,
            code=self._code,
            blocksize=self.block_size,
            quant_type='nf4',
            dtype=self.compute_dtype,
            offset=self._offset,
            state2=bnb.functional.QuantState(
                absmax=self._nested_absmax,
                shape=None,
                code=self._nested_code,
                blocksize=self.nested_block_size,
                quant_type='nf4',
                dtype=torch.float32,
                offset=None,
                state2=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(dtype=self.compute_dtype)
        x = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.quant_state)
        return x.to(dtype=orig_dtype)
