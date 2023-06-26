import os
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image
from tqdm import tqdm

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

        # # diffusion model
        # tokenizer = self.model.tokenizer
        # text_encoder = self.model.text_encoder
        # noise_scheduler = self.model.noise_scheduler
        # unet = self.model.unet
        # movq = self.model.movq
        # movq_scale_factor = self.diffusion_pipeline.movq_scale_factor
        #
        # height, width = resolution
        #
        # # prepare prompt
        # tokenizer_output = tokenizer(
        #     prompt,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=tokenizer.model_max_length,
        #     return_tensors="pt",
        # )
        # tokens = tokenizer_output.input_ids.to(self.train_device)
        # if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        #     tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)
        # else:
        #     tokens_attention_mask = None
        #
        # negative_tokenizer_output = tokenizer(
        #     "",
        #     padding='max_length',
        #     truncation=True,
        #     max_length=tokenizer.model_max_length,
        #     return_tensors="pt",
        # )
        # negative_tokens = negative_tokenizer_output.input_ids.to(self.train_device)
        # if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        #     negative_tokens_attention_mask = negative_tokenizer_output.attention_mask.to(self.train_device)
        # else:
        #     negative_tokens_attention_mask = None
        #
        # text_encoder_output = text_encoder(
        #     tokens,
        #     attention_mask=tokens_attention_mask,
        #     return_dict=True
        # )
        # prompt_embedding = text_encoder_output.last_hidden_state
        #
        # text_encoder_output = text_encoder(
        #     negative_tokens,
        #     attention_mask=negative_tokens_attention_mask,
        #     return_dict=True
        # )
        # negative_prompt_embedding = text_encoder_output.last_hidden_state
        #
        # combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])
        #
        # # prepare timesteps
        # noise_scheduler.set_timesteps(steps, device=self.train_device)
        # timesteps = noise_scheduler.timesteps
        #
        # # prepare latent image
        # num_channels_latents = unet.config.in_channels
        # latent_image = torch.randn(
        #     size=(1, num_channels_latents, height // movq_scale_factor, width // movq_scale_factor),
        #     generator=generator,
        #     device=self.train_device,
        #     dtype=torch.float32
        # )
        #
        # # denoising loop
        # for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
        #     latent_model_input = torch.cat([latent_image] * 2)
        #     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)
        #
        #     # predict the noise residual
        #     added_cond_kwargs = {"text_embeds": prompt_embeds, "image_embeds": image_embeds}
        #     noise_pred = unet(
        #         sample=latent_model_input,
        #         timestep=timestep,
        #         encoder_hidden_states=text_encoder_hidden_states,
        #         added_cond_kwargs=added_cond_kwargs,
        #         return_dict=False,
        #     )[0]
        #
        #     # cfg
        #     noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
        #     noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)
        #
        #     # compute the previous noisy sample x_t -> x_t-1
        #     latent_image = noise_scheduler.step(
        #         noise_pred, timestep, latent_image, return_dict=False
        #     )[0]
        #
        # image = movq.decode(latent_image, return_dict=False)[0]
        #
        # return image[0]

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
