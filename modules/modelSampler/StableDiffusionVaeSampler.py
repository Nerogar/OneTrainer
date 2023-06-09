import os
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import transforms

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ModelType import ModelType


class StableDiffusionVaeSampler(BaseModelSampler):
    def __init__(self, model: StableDiffusionModel, model_type: ModelType, train_device: torch.device):
        self.model = model
        self.model_type = model_type
        self.train_device = train_device

    def sample(
            self,
            prompt: str,
            resolution: tuple[int, int],
            seed: int,
            destination: str,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        # TODO: this is reusing the prompt parameters as the image path, think of a better solution

        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)

        image = Image.open(prompt)
        image = image.convert("RGB")

        t_in = transforms.ToTensor()
        image_tensor = t_in(image).to(device=self.train_device, dtype=self.model.vae.dtype)
        image_tensor = image_tensor * 2 - 1

        with torch.no_grad():
            latent_image_tensor = self.model.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean
            image_tensor = self.model.vae.decode(latent_image_tensor).sample.squeeze()

        image_tensor = (image_tensor + 1) * 0.5
        image_tensor = image_tensor.clamp(0, 1)

        t_out = transforms.ToPILImage()
        image = t_out(image_tensor)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)
