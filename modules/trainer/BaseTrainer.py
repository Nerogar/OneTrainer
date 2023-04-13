import time
from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.TimeUnit import TimeUnit


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args: TrainArgs):
        self.args = args
        self.previous_action = {}

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

    @staticmethod
    def __masked_loss(
            loss_fn: Callable,
            predicted: Tensor,
            target: Tensor,
            mask: Tensor,
            unmasked_weight: float,
            normalize_masked_area_loss: bool
    ) -> Tensor:
        clamped_mask = torch.clamp(mask, unmasked_weight, 1)

        masked_predicted = predicted * clamped_mask
        masked_target = target * clamped_mask

        losses = loss_fn(masked_predicted, masked_target, reduction="none")

        if normalize_masked_area_loss:
            losses = losses / clamped_mask.mean(dim=(1, 2, 3))

        del clamped_mask

        return losses

    def loss(self, batch: dict, predicted: Tensor, target: Tensor) -> Tensor:
        losses = None
        if self.args.masked_training and not self.args.model_type.has_conditioning_image_input():
            match self.args.loss_function:
                case LossFunction.MSE:
                    losses = self.__masked_loss(
                        F.mse_loss,
                        predicted,
                        target,
                        batch['latent_mask'],
                        self.args.unmasked_weight,
                        self.args.normalize_masked_area_loss
                    ).mean([1, 2, 3])
        else:
            match self.args.loss_function:
                case LossFunction.MSE:
                    losses = F.mse_loss(
                        predicted,
                        target,
                        reduction='none'
                    ).mean([1, 2, 3])

            if self.args.normalize_masked_area_loss:
                clamped_mask = torch.clamp(batch['latent_mask'], self.args.unmasked_weight, 1)
                losses = losses / clamped_mask.mean(dim=(1, 2, 3))

        return losses.mean()

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.args.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(
            self.args.model_type,
            self.args.train_device,
            self.args.temp_device,
            self.args.training_method,
            self.args.debug_mode,
        )

    def create_data_loader(self, model: BaseModel, train_progress: TrainProgress):
        return create.create_data_loader(
            model,
            self.args.model_type,
            self.args.training_method,
            self.args,
            train_progress,
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.args.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(model, self.args.model_type, self.args.train_device)

    def action_needed(self, name: str, interval: float, unit: TimeUnit, train_progress: TrainProgress, start_at_zero: bool = True):
        if name not in self.previous_action:
            self.previous_action[name] = -1

        match unit:
            case TimeUnit.EPOCH:
                if start_at_zero:
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0
                else:
                    # should actually be the last step of each epoch, but we don't know how many steps an epoch has
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0 and train_progress.epoch > 0
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
