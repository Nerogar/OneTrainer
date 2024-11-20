from abc import ABCMeta, abstractmethod


class QuantizedLinearMixin(metaclass=ABCMeta):
    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]:
        pass
