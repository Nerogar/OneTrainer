from abc import ABCMeta, abstractmethod

import torch
from torch.optim import Optimizer

from modules.model.BaseModel import BaseModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class BaseModelSetup(metaclass=ABCMeta):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode

    @abstractmethod
    def setup_model(
            self,
            model: BaseModel,
            args: TrainArgs,
    ):
        pass

    @abstractmethod
    def setup_eval_device(
            self,
            model: BaseModel,
    ):
        pass

    @abstractmethod
    def setup_train_device(
            self,
            model: BaseModel,
            args: TrainArgs,
    ):
        pass

    @abstractmethod
    def get_optimizer(
            self,
            model: BaseModel,
            args: TrainArgs,
    ) -> Optimizer:
        pass

    @abstractmethod
    def get_train_progress(
            self,
            model: BaseModel,
            args: TrainArgs,
    ) -> TrainProgress:
        pass
