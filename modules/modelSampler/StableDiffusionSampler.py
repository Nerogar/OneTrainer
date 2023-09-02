import inspect
import os
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image
from tqdm import tqdm

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.params.SampleParams import SampleParams


class StableDiffusionSampler(BaseModelSampler):
    def __init__(self, model: StableDiffusionModel, model_type: ModelType, train_device: torch.device):
        self.model = model
        self.model_type = model_type
        self.train_device = train_device
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            height: int,
            width: int,
            seed: int,
            steps: int,
            cfg_scale: float,
            cfg_rescale: float = 0.7,
            text_encoder_layer_skip: int = 0,
            force_last_timestep: bool = False,
    ) -> Image:
        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)
        tokenizer = self.pipeline.tokenizer
        text_encoder = self.pipeline.text_encoder
        noise_scheduler = self.pipeline.scheduler
        image_processor = self.pipeline.image_processor
        unet = self.pipeline.unet
        vae = self.pipeline.vae
        vae_scale_factor = self.pipeline.vae_scale_factor

        # prepare prompt
        tokenizer_output = tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = tokenizer_output.input_ids.to(self.train_device)
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)
        else:
            tokens_attention_mask = None

        negative_tokenizer_output = tokenizer(
            "",
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        negative_tokens = negative_tokenizer_output.input_ids.to(self.train_device)
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            negative_tokens_attention_mask = negative_tokenizer_output.attention_mask.to(self.train_device)
        else:
            negative_tokens_attention_mask = None

        with torch.autocast(self.train_device.type):
            if text_encoder_layer_skip > 0:
                text_encoder_output = text_encoder(
                    tokens,
                    return_dict=True,
                    output_hidden_states=True,
                )
                final_layer_norm = text_encoder.text_model.final_layer_norm
                prompt_embedding = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
                )

                text_encoder_output = text_encoder(
                    negative_tokens,
                    return_dict=True,
                    output_hidden_states=True,
                )
                final_layer_norm = text_encoder.text_model.final_layer_norm
                negative_prompt_embedding = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
                )
            else:
                text_encoder_output = text_encoder(
                    tokens,
                    attention_mask=tokens_attention_mask,
                    return_dict=True,
                )
                prompt_embedding = text_encoder_output.last_hidden_state

                text_encoder_output = text_encoder(
                    negative_tokens,
                    attention_mask=negative_tokens_attention_mask,
                    return_dict=True,
                )
                negative_prompt_embedding = text_encoder_output.last_hidden_state

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])

        # prepare timesteps
        noise_scheduler.set_timesteps(steps, device=self.train_device)
        timesteps = noise_scheduler.timesteps

        if force_last_timestep:
            last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                            * (noise_scheduler.config.num_train_timesteps - 1)

            # add the final timestep to force predicting with zero snr
            timesteps = torch.cat([last_timestep, timesteps])

        # prepare latent image
        num_channels_latents = unet.config.in_channels
        latent_image = torch.randn(
            size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32
        ) * noise_scheduler.init_noise_sigma

        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        # denoising loop
        for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
            latent_model_input = torch.cat([latent_image] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            with torch.autocast(self.train_device.type):
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=combined_prompt_embedding,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

            # cfg
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            if cfg_rescale > 0.0:
                # From: Common Diffusion Noise Schedules and Sample Steps are Flawed (https://arxiv.org/abs/2305.08891)
                std_positive = noise_pred_positive.std(dim=list(range(1, noise_pred_positive.ndim)), keepdim=True)
                std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                noise_pred_rescaled = noise_pred * (std_positive / std_pred)
                noise_pred = (
                        cfg_rescale * noise_pred_rescaled + (1 - cfg_rescale) * noise_pred
                )

            # compute the previous noisy sample x_t -> x_t-1
            latent_image = noise_scheduler.step(
                noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
            )[0]

        latent_image = latent_image.to(dtype=vae.dtype)
        image = vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

        return image[0]

    @torch.no_grad()
    def __sample_inpainting(
            self,
            prompt: str,
            height: int,
            width: int,
            seed: int,
            steps: int,
            cfg_scale: float,
            cfg_rescale: float = 0.7,
            text_encoder_layer_skip: int = 0,
            force_last_timestep: bool = False,
    ) -> Image:
        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(seed)
        tokenizer = self.pipeline.tokenizer
        text_encoder = self.pipeline.text_encoder
        noise_scheduler = self.pipeline.scheduler
        image_processor = self.pipeline.image_processor
        unet = self.pipeline.unet
        vae = self.pipeline.vae
        vae_scale_factor = self.pipeline.vae_scale_factor

        # prepare conditioning image
        conditioning_image = torch.zeros((1, 3, height, width), dtype=torch.float32, device=self.train_device)
        conditioning_image = conditioning_image
        latent_conditioning_image = vae.encode(conditioning_image).latent_dist.mode() * vae.config.scaling_factor
        latent_mask = torch.ones(
            size=(1, 1, latent_conditioning_image.shape[2], latent_conditioning_image.shape[3]),
            dtype=torch.float32,
            device=self.train_device
        )

        # prepare prompt
        tokenizer_output = tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = tokenizer_output.input_ids.to(self.train_device)
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)
        else:
            tokens_attention_mask = None

        negative_tokenizer_output = tokenizer(
            "",
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        negative_tokens = negative_tokenizer_output.input_ids.to(self.train_device)
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            negative_tokens_attention_mask = negative_tokenizer_output.attention_mask.to(self.train_device)
        else:
            negative_tokens_attention_mask = None

        with torch.autocast(self.train_device.type):
            if text_encoder_layer_skip > 0:
                text_encoder_output = text_encoder(
                    tokens,
                    return_dict=True,
                    output_hidden_states=True,
                )
                final_layer_norm = text_encoder.text_model.final_layer_norm
                prompt_embedding = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
                )

                text_encoder_output = text_encoder(
                    negative_tokens,
                    return_dict=True,
                    output_hidden_states=True,
                )
                final_layer_norm = text_encoder.text_model.final_layer_norm
                negative_prompt_embedding = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
                )
            else:
                text_encoder_output = text_encoder(
                    tokens,
                    attention_mask=tokens_attention_mask,
                    return_dict=True,
                )
                prompt_embedding = text_encoder_output.last_hidden_state

                text_encoder_output = text_encoder(
                    negative_tokens,
                    attention_mask=negative_tokens_attention_mask,
                    return_dict=True,
                )
                negative_prompt_embedding = text_encoder_output.last_hidden_state

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])

        # prepare timesteps
        noise_scheduler.set_timesteps(steps, device=self.train_device)
        timesteps = noise_scheduler.timesteps

        if force_last_timestep:
            last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                            * (noise_scheduler.config.num_train_timesteps - 1)

            # add the final timestep to force predicting with zero snr
            timesteps = torch.cat([last_timestep, timesteps])

        # prepare latent image
        num_channels_latents = latent_conditioning_image.shape[1]
        latent_image = torch.randn(
            size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32
        ) * noise_scheduler.init_noise_sigma

        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        # denoising loop
        for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
            latent_model_input = torch.concat(
                [latent_image, latent_mask, latent_conditioning_image], 1
            )
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            with torch.autocast(self.train_device.type):
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=combined_prompt_embedding,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

            # cfg
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            if cfg_rescale > 0.0:
                # From: Common Diffusion Noise Schedules and Sample Steps are Flawed (https://arxiv.org/abs/2305.08891)
                std_positive = noise_pred_positive.std(dim=list(range(1, noise_pred_positive.ndim)), keepdim=True)
                std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                noise_pred_rescaled = noise_pred * (std_positive / std_pred)
                noise_pred = (
                        cfg_rescale * noise_pred_rescaled + (1 - cfg_rescale) * noise_pred
                )

            # compute the previous noisy sample x_t -> x_t-1
            latent_image = noise_scheduler.step(
                noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
            )[0]

        latent_image = latent_image.to(dtype=vae.dtype)
        image = vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

        return image[0]

    def sample(
            self,
            sample_params: SampleParams,
            destination: str,
            image_format: ImageFormat,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        prompt = sample_params.prompt

        if len(self.model.embeddings) > 0:
            tokens = [f"<embedding_{i}>" for i in range(self.model.embeddings[0].token_count)]
            embedding_string = ''.join(tokens)
            prompt = prompt.replace("<embedding>", embedding_string)

        if self.model_type.has_conditioning_image_input():
            image = self.__sample_inpainting(
                prompt=prompt,
                height=sample_params.height,
                width=sample_params.width,
                seed=sample_params.seed,
                steps=20,
                cfg_scale=7,
                cfg_rescale=0.7 if force_last_timestep else 0.0,
                text_encoder_layer_skip=text_encoder_layer_skip,
                force_last_timestep=force_last_timestep
            )
        else:
            image = self.__sample_base(
                prompt=prompt,
                height=sample_params.height,
                width=sample_params.width,
                seed=sample_params.seed,
                steps=20,
                cfg_scale=7,
                cfg_rescale=0.7 if force_last_timestep else 0.0,
                text_encoder_layer_skip=text_encoder_layer_skip,
                force_last_timestep=force_last_timestep
            )

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)

        on_sample(image)
