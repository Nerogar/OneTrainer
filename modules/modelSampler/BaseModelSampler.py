from abc import ABCMeta, abstractmethod
from collections.abc import Callable

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat

import torch

from PIL.Image import Image


class BaseModelSampler(metaclass=ABCMeta):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            on_sample: Callable[[Image], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        pass

    @staticmethod
    def quantize_resolution(resolution: int, quantization: int) -> int:
        return round(resolution / quantization) * quantization
