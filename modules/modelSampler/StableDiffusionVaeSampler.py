import os
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import transforms

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.config.SampleParams import SampleConfig


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
            sample_params: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: Callable[[Image], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # TODO: this is reusing the prompt parameters as the image path, think of a better solution
        image = Image.open(sample_params.prompt)
        image = image.convert("RGB")

        t_in = transforms.ToTensor()
        image_tensor = t_in(image).to(device=self.train_device, dtype=self.model.vae.dtype)
        image_tensor = image_tensor * 2 - 1

        self.model.vae_to(self.train_device)

        with torch.no_grad():
            latent_image_tensor = self.model.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean
            image_tensor = self.model.vae.decode(latent_image_tensor).sample.squeeze()

        self.model.vae_to(self.temp_device)

        image_tensor = (image_tensor + 1) * 0.5
        image_tensor = image_tensor.clamp(0, 1)

        t_out = transforms.ToPILImage()
        image = t_out(image_tensor)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)
