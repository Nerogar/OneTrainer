from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class BaseModelLoader(metaclass=ABCMeta):
    @staticmethod
    def _create_default_model_spec(
            model_type: ModelType,
    ) -> ModelSpec:
        return ModelSpec()

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> BaseModel | None:
        pass
