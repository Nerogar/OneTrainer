from abc import ABCMeta, abstractmethod
from typing import Callable

from PIL.Image import Image


class BaseModelSampler(metaclass=ABCMeta):

    @abstractmethod
    def sample(
            self,
            prompt: str,
            resolution: tuple[int, int],
            seed: int,
            destination: str,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        pass
