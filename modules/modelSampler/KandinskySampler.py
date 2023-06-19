import os
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image

from modules.model.KandinskyModel import KandinskyModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ModelType import ModelType


class KandinskySampler(BaseModelSampler):
    def __init__(self, model: KandinskyModel, model_type: ModelType, train_device: torch.device):
        self.model = model
        self.model_type = model_type
        self.train_device = train_device
        self.prior_pipeline = self.model.create_prior_pipeline()
        self.diffusion_pipeline = model.create_diffusion_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            resolution: tuple[int, int],
            seed: int,
            prior_steps: int,
            steps: int,
            prior_cfg_scale: float,
            cfg_scale: float,
    ) -> Image:
        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)

        image_embeds, negative_image_embeds = self.prior_pipeline(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=prior_steps,
            guidance_scale=prior_cfg_scale,
            generator=generator,
        ).to_tuple()

        image = self.diffusion_pipeline(
            prompt=prompt,
            negative_prompt="",
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=resolution[0],
            width=resolution[1],
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
        ).images[0]

        return image

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
            prior_steps=10,
            steps=20,
            prior_cfg_scale=4,
            cfg_scale=2,
        )

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)

        on_sample(image)
