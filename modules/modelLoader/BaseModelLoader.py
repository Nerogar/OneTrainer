from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self, base_model_name: str, model_type: ModelType, dtype: torch.dtype) -> BaseModel | None:
        pass
