from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class BaseModelSaver(metaclass=ABCMeta):

    @abstractmethod
    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        pass
