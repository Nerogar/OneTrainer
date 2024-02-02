from abc import ABCMeta, abstractmethod
from typing import Iterable, Iterator

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.BaseModel import BaseModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScaler import LearningRateScaler


class BaseModelSetup(metaclass=ABCMeta):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(BaseModelSetup, self).__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode

    @abstractmethod
    def create_parameters(
            self,
            model: BaseModel,
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        pass

    def create_parameters_for_optimizer(
            self,
            model: BaseModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        return self.create_parameters(model, config)

    @abstractmethod
    def setup_model(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_train_device(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass

    def create_param_groups(
            self,
            config: TrainConfig,
            params: Iterator[Parameter] | list[Parameter],
            lr_arg: float,
    ) -> dict:
        batch_size = 1 if config.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.GRADIENT_ACCUMULATION] else config.batch_size
        gradient_accumulation_steps = 1 if config.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.BATCH] else config.gradient_accumulation_steps

        # Determine the learning rate
        lr = lr_arg if lr_arg is not None else config.learning_rate
        lr = lr * ((batch_size * gradient_accumulation_steps) ** 0.5)

        # Create a parameter group for the text encoder
        return {
            'params': list(params),
            'lr': lr,
            'initial_lr': lr,
        }
