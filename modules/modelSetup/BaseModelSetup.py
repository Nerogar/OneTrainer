import os
from abc import ABCMeta, abstractmethod
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter
from torchvision import transforms

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

    def save_image(self, image_tensor: Tensor, directory: str, name: str, step: int):
        path = os.path.join(directory, "step-" + str(step) + "-" + name + ".png")
        if not os.path.exists(directory):
            os.makedirs(directory)

        t = transforms.ToPILImage()

        image_tensor = image_tensor[0].unsqueeze(0)

        range_min = -1
        range_max = 1
        image_tensor = (image_tensor - range_min) / (range_max - range_min)
        image_tensor = image_tensor.clamp(0.0, 1.0)

        image = t(image_tensor.squeeze())
        image.save(path)

    def project_latent_to_image(self, latent_tensor: Tensor):
        generator = torch.Generator(device=latent_tensor.device)
        generator.manual_seed(42)
        weight = torch.randn((3, 4, 1, 1), generator=generator, device=latent_tensor.device, dtype=latent_tensor.dtype)

        with torch.no_grad():
            return torch.nn.functional.conv2d(latent_tensor, weight) / 3.0

    @abstractmethod
    def create_parameters(
            self,
            model: BaseModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        pass

    def create_parameters_for_optimizer(
            self,
            model: BaseModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return self.create_parameters(model, args)

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
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
            self,
            model: BaseModel,
            args: TrainArgs,
            train_progress: TrainProgress,
    ):
        pass
