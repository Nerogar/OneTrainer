from abc import ABCMeta, abstractmethod

import torch
from torch.optim import Optimizer

from modules.module.EMAModule import EMAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class BaseModel(metaclass=ABCMeta):
    model_type: ModelType
    optimizer: Optimizer | None
    optimizer_state_dict: dict | None
    ema: EMAModuleWrapper
    ema_state_dict: dict | None
    train_progress: TrainProgress
    model_spec: ModelSpec | None
    train_config: TrainConfig | None

    def __init__(
            self,
            model_type: ModelType,
            optimizer_state_dict: dict | None,
            ema_state_dict: dict | None,
            train_progress: TrainProgress,
            model_spec: ModelSpec | None,
            train_config: TrainConfig | None,
    ):
        self.model_type = model_type
        self.optimizer = None
        self.optimizer_state_dict = optimizer_state_dict
        self.ema_state_dict = ema_state_dict
        self.train_progress = train_progress if train_progress is not None else TrainProgress()
        self.model_spec = model_spec
        self.train_config = train_config

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def eval(self):
        pass
