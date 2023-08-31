from abc import ABCMeta, abstractmethod
from typing import Callable

from PIL.Image import Image

from modules.util.params.SampleParams import SampleParams


class BaseModelSampler(metaclass=ABCMeta):

    @abstractmethod
    def sample(
            self,
            sample_params: SampleParams,
            destination: str,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        pass
