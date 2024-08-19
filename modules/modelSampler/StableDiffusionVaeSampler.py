import os
from pathlib import Path
from typing import Callable

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType

import torch
from torchvision.transforms import transforms

from PIL import Image


class StableDiffusionVaeSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusionModel,
            model_type: ModelType,
    ):
        super(StableDiffusionVaeSampler, self).__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            on_sample: Callable[[Image], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # TODO: this is reusing the prompt parameters as the image path, think of a better solution
        image = Image.open(sample_config.prompt)
        image = image.convert("RGB")
        # TODO: figure out better set of transformations for resize and/or implement way to configure them as per-sample toggle
        scale = sample_config.height
        if sample_config.width > sample_config.height:
            sample_config.width
        
        t_in = transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop([sample_config.height, sample_config.width]),
            transforms.ToTensor()
        ])
        image_tensor = t_in(image).to(device=self.train_device, dtype=self.model.vae.dtype)
        image_tensor = image_tensor * 2 - 1

        self.model.vae_to(self.train_device)

        with torch.no_grad():
            latent_image_tensor = self.model.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean
            image_tensor = self.model.vae.decode(latent_image_tensor).sample.clamp(-1, 1).squeeze()

        self.model.vae_to(self.temp_device)

        t_out = transforms.ToPILImage()
        image = t_out(image_tensor)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)
        
        on_sample(image)
