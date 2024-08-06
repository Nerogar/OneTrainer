from abc import ABCMeta, abstractmethod

from modules.module.EMAModule import EMAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress

import torch
from torch.optim import Optimizer


class BaseModel(metaclass=ABCMeta):
    model_type: ModelType
    parameters: NamedParameterGroupCollection | None
    optimizer: Optimizer | None
    optimizer_state_dict: dict | None
    param_group_mapping: list[str] | None
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
        self.parameters = None
        self.optimizer = None
        self.optimizer_state_dict = optimizer_state_dict
        self.param_group_mapping = None
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
