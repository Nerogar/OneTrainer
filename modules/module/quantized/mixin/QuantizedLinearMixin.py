from abc import ABCMeta, abstractmethod

import torch


class QuantizedLinearMixin(metaclass=ABCMeta):
    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]:
        pass

    # 'device' only says where to run the dequantization, like in quantize(); the device of the returned
    # tensor is unspecified, so callers needing it somewhere specific have to move it themselves.
    # Layer offloading relocates only 'weight' and 'bias', so separate quantization state (scales, absmax)
    # can be left behind on another device and has to be brought back together here.
    @abstractmethod
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass
