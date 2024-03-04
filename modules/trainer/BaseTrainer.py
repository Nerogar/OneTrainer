import time
from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands


class BaseTrainer(
    TimedActionMixin,
    metaclass=ABCMeta,
):
    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.callbacks = callbacks
        self.commands = commands
        self.train_device = torch.device(self.config.train_device)
        self.temp_device = torch.device(self.config.temp_device)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def backup(self, train_progress: TrainProgress):
        pass

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.config.model_type, self.config.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(
            self.config.model_type,
            self.train_device,
            self.temp_device,
            self.config.training_method,
            self.config.debug_mode,
        )

    def create_data_loader(self, model: BaseModel, train_progress: TrainProgress):
        return create.create_data_loader(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            self.config.training_method,
            self.config,
            train_progress,
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.config.model_type, self.config.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            self.config.training_method
        )

