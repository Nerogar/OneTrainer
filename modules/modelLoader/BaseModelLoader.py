from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            weight_dtype: torch.dtype,
            base_model_name: str,
            extra_model_name: str | None
    ) -> BaseModel | None:
        pass
