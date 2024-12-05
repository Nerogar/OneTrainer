from abc import ABCMeta, abstractmethod

import torch


class QuantizedLinearMixin(metaclass=ABCMeta):
    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass
