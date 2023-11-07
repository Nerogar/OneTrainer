from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> BaseModel | None:
        pass
