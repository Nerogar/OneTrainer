from abc import ABCMeta, abstractmethod

import torch


class LoRAFusableLinearMixin(metaclass=ABCMeta):
    @abstractmethod
    def forward_with_lora(self, x: torch.Tensor, lora_down: torch.Tensor, lora_up: torch.Tensor, dropout: torch.nn.Dropout, alpha: torch.Tensor) -> torch.Tensor:
        pass
