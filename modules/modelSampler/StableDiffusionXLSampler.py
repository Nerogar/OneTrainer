import os
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ModelType import ModelType


class StableDiffusionXLSampler(BaseModelSampler):
    def __init__(self, model: StableDiffusionXLModel, model_type: ModelType, train_device: torch.device):
        self.model = model
        self.model_type = model_type
        self.train_device = train_device
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            resolution: tuple[int, int],
            seed: int,
            steps: int,
            cfg_scale: float,
            cfg_rescale: float = 0.7,
            text_encoder_layer_skip: int = 0,
            force_last_timestep: bool = False,
    ) -> Image:
        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)

        height, width = resolution

        images = self.pipeline(
            generator=generator,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images

        return images[0]

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
        image = self.__sample_base(
            prompt=prompt,
            resolution=resolution,
            seed=seed,
            steps=20,
            cfg_scale=7,
            cfg_rescale=0.7 if force_last_timestep else 0.0,
            text_encoder_layer_skip=text_encoder_layer_skip,
            force_last_timestep=force_last_timestep
        )

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)

        on_sample(image)
