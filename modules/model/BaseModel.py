from abc import ABCMeta, abstractmethod
from typing import Iterator

from torch import Tensor
from torch.nn import Parameter

from modules.util.args.TrainArgs import TrainArgs


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def parameters(self, args: TrainArgs) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def predict(self, batch: dict, args: TrainArgs, step: int) -> (Tensor, Tensor):
        pass
