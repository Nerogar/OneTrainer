from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class ModelSetupText2ImageMixin(
    metaclass=ABCMeta,
):
    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            seed: int,
            *,
            #extend signature of BaseModelSetup.predict() by one parameter only known to image models:
            timestep: Tensor = None,
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

    @torch.no_grad()
    def calculate_validation_losses(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> dict:
        losses = {}
        tasks = [int(x) for x in config.validation_timesteps.split(',')]
        for task in tasks:
            timestep=torch.tensor([task], dtype=torch.long, device=config.train_device)
            model_output_data = self.predict(model, batch, config, train_progress, seed=0, timestep=timestep)
            loss = self.calculate_loss(model, batch, model_output_data, config)
            losses[task] = loss.item()
        return losses
