from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> BaseModel | None:
        pass
