import os
from pathlib import Path

import torch

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ModelType import ModelType


class StableDiffusionSampler(BaseModelSampler):
    def __init__(self, model: StableDiffusionModel, model_type: ModelType, train_device: torch.device):
        self.model = model
        self.model_type = model_type
        self.train_device = train_device
        self.pipeline = model.create_pipeline()

    def sample(self, prompt: str, resolution: tuple[int, int], seed: int, destination: str):
        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)

        if len(self.model.embeddings) > 0:
            tokens = [f"<embedding_{i}>" for i in range(self.model.embeddings[0].token_count)]
            embedding_string = ''.join(tokens)
            prompt = prompt.replace("<embedding>", embedding_string)

        if self.model_type.has_conditioning_image_input():
            conditioning_image = torch.zeros(size=(3, resolution[0], resolution[1]))
            mask_image = torch.ones(size=(1, resolution[0], resolution[1]))

            output = self.pipeline(
                prompt=prompt,
                image=conditioning_image,
                mask_image=mask_image,
                height=resolution[0],
                width=resolution[1],
                num_inference_steps=20,
                guidance_scale=7,
                num_images_per_prompt=1,
                return_dict=True,
                generator=generator,
            )
        else:
            output = self.pipeline(
                prompt=prompt,
                height=resolution[0],
                width=resolution[1],
                num_inference_steps=20,
                guidance_scale=7,
                num_images_per_prompt=1,
                return_dict=True,
                generator=generator,
            )

        image = output.images[0]

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)
