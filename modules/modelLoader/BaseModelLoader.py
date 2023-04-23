from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self, model_type: ModelType, base_model_name: str, extra_model_name: str | None) -> BaseModel | None:
        pass
