from abc import ABCMeta, abstractmethod

import torch


class QuantizedModuleMixin(metaclass=ABCMeta):
    @abstractmethod
    def quantize(self, device: torch.device | None = None):
        pass
