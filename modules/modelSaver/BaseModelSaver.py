from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class BaseModelSaver(metaclass=ABCMeta):

    @staticmethod
    def _convert_state_dict_dtype(state_dict: dict, dtype: torch.dtype) -> dict:
        converted_state_dict = {}

        for (key, value) in state_dict.items():
            if isinstance(value, dict):
                converted_state_dict[key] = BaseModelSaver._convert_state_dict_dtype(value, dtype)
            else:
                converted_state_dict[key] = value.to(dtype=dtype)

        return converted_state_dict

    @staticmethod
    def _convert_state_dict_to_contiguous(state_dict: dict):
        for (key, value) in state_dict.items():
            if isinstance(value, dict):
                BaseModelSaver._convert_state_dict_to_contiguous(value)
            else:
                state_dict[key] = value.contiguous()

    @abstractmethod
    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        pass
