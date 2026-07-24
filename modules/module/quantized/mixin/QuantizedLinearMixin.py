from abc import ABCMeta, abstractmethod

import torch


class QuantizedLinearMixin(metaclass=ABCMeta):
    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass

    @abstractmethod
    def mark_needs_requantization(self):
        # reset the concrete class's is-quantized flag so the next materialize re-quantizes. Called by streaming
        # eviction, which discards the packed weights back to meta.
        pass

    def predict_offload_bytes(self) -> int:
        # post-quantization offload footprint, predicted from the unpacked skeleton shape while the module is still a
        # meta skeleton (the real packed tensors don't exist yet).
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_offload_bytes (disk-offload conductor sizing)")
