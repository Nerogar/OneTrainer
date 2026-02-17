import os
import subprocess
import sys
from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch


class BaseTrainer(
    TimedActionMixin,
    metaclass=ABCMeta,
):

    tensorboard_subprocess: subprocess.Popen

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__()
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

    def create_data_loader(self, model: BaseModel, model_setup: BaseModelSetup, train_progress: TrainProgress, is_validation=False):
        return create.create_data_loader(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            model_setup,
            self.config.training_method,
            self.config,
            train_progress,
            is_validation,
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
    def _start_tensorboard(self):
        tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        tensorboard_log_dir = os.path.join(self.config.workspace_dir, "tensorboard")

        tensorboard_args = [
            tensorboard_executable,
            "--logdir",
            tensorboard_log_dir,
            "--port",
            str(self.config.tensorboard_port),
            "--samples_per_plugin=images=100,scalars=10000",
        ]

        if self.config.tensorboard_expose:
            tensorboard_args.append("--bind_all")

        self.tensorboard_subprocess = subprocess.Popen(tensorboard_args)

    def _stop_tensorboard(self):
        self.tensorboard_subprocess.kill()
