import time
from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.TimeUnit import TimeUnit


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args: TrainArgs, callbacks: TrainCallbacks, commands: TrainCommands):
        self.args = args
        self.callbacks = callbacks
        self.commands = commands
        self.previous_action = {}
        self.train_device = torch.device(self.args.train_device)
        self.temp_device = torch.device(self.args.temp_device)

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
    def backup(self):
        pass

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.args.model_type, self.args.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(
            self.args.model_type,
            self.train_device,
            self.temp_device,
            self.args.training_method,
            self.args.debug_mode,
        )

    def create_data_loader(self, model: BaseModel, train_progress: TrainProgress):
        return create.create_data_loader(
            self.train_device,
            self.temp_device,
            model,
            self.args.model_type,
            self.args.training_method,
            self.args,
            train_progress,
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.args.model_type, self.args.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            self.train_device,
            self.temp_device,
            model,
            self.args.model_type,
            self.args.training_method
        )

    def action_needed(self, name: str, interval: float, unit: TimeUnit, train_progress: TrainProgress,
                      start_at_zero: bool = True):
        if name not in self.previous_action:
            self.previous_action[name] = -1

        match unit:
            case TimeUnit.EPOCH:
                if start_at_zero:
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0
                else:
                    # should actually be the last step of each epoch, but we don't know how many steps an epoch has
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0 \
                        and train_progress.epoch > 0
            case TimeUnit.STEP:
                if start_at_zero:
                    return train_progress.global_step % int(interval) == 0
                else:
                    return (train_progress.global_step + 1) % int(interval) == 0
            case TimeUnit.SECOND:
                if not start_at_zero and self.previous_action[name] < 0:
                    self.previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.previous_action[name]
                if seconds_since_previous_action > interval:
                    self.previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.MINUTE:
                if not start_at_zero and self.previous_action[name] < 0:
                    self.previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.previous_action[name]
                if seconds_since_previous_action > (interval * 60):
                    self.previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.HOUR:
                if not start_at_zero and self.previous_action[name] < 0:
                    self.previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.previous_action[name]
                if seconds_since_previous_action > (interval * 60 * 60):
                    self.previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.NEVER:
                return False
            case TimeUnit.ALWAYS:
                return True
            case _:
                return False
