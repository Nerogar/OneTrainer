from abc import ABCMeta, abstractmethod
from typing import Iterator

from torch import Tensor
from torch.nn import Parameter

from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.ModelType import ModelType


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_type: ModelType):
        self.model_type = model_type

    @abstractmethod
    def parameters(self, args: TrainArgs) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def predict(self, batch: dict, args: TrainArgs) -> (Tensor, Tensor):
        pass
