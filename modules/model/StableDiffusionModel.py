from typing import Iterator

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torch import Tensor
from torch.nn import Parameter
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.BaseModel import BaseModel
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionModel(BaseModel):
    def __init__(
            self,
            tokenizer: CLIPTokenizer,
            noise_scheduler: DDPMScheduler,
            text_encoder: CLIPTextModel,
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            image_depth_processor: DPTImageProcessor,
            depth_estimator: DPTForDepthEstimation,
    ):
        super(StableDiffusionModel, self).__init__()

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

    def parameters(self, args: TrainArgs) -> Iterator[Parameter]:
        if args.train_text_encoder:
            return list(self.text_encoder.parameters()) + list(self.unet.parameters())
        else:
            return list(self.unet.parameters())

    def predict(self, batch: dict, args: TrainArgs) -> (Tensor, Tensor):
        latent_image = batch['latent_image']
        latent_conditioning_image = batch['latent_conditioning_image']

        latent_image = latent_image * self.vae.scaling_factor
        latent_conditioning_image = latent_conditioning_image * self.vae.scaling_factor

        if args.offset_noise_weight > 0:
            noise = torch.randn_like(latent_image) + (args.offset_noise_weight * torch.randn(latent_image.shape[0], latent_image.shape[1], 1, 1).to(latent_image.device))
        else:
            noise = torch.randn_like(latent_image)
        timestep = torch.randint(0, self.noise_scheduler.config['num_train_timesteps'], (latent_image.shape[0],), device=latent_image.device).long()

        noisy_latent_image = self.noise_scheduler.add_noise(original_samples=latent_image, noise=noise, timesteps=timestep)

        text_encoder_output = self.text_encoder(batch['tokens'])[0]

        if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
            latent_input = torch.concat([noisy_latent_image, batch['latent_mask'], latent_conditioning_image], 1)
        else:
            latent_input = noisy_latent_image

        if args.model_type.has_depth_input():
            return self.unet(latent_input, timestep, text_encoder_output, batch['latent_depth']).sample, noise
        else:
            return self.unet(latent_input, timestep, text_encoder_output).sample, noise
