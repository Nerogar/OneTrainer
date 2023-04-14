from abc import ABCMeta

from modules.util.enum.ModelType import ModelType


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
