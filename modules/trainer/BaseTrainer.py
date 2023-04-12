from abc import ABCMeta
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
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LossFunction import LossFunction


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args: TrainArgs):
        self.args = args

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
        if self.args.masked_training:
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

    def create_data_loader(self, model: BaseModel):
        return create.create_data_loader(
            model,
            self.args.model_type,
            self.args.training_method,
            self.args
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.args.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(model, self.args.model_type, self.args.train_device)
