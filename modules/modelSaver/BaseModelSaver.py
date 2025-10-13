from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class BaseModelSaver(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _ensure_correct_extension(
            self,
            output_model_format: ModelFormat,
            output_model_destination: str,
    ) -> str:

        if output_model_format.is_single_file():
            expected_extension = output_model_format.file_extension()
            if not output_model_destination.endswith(expected_extension):
                return output_model_destination + expected_extension

        return output_model_destination

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
