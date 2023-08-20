import os
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image
from tqdm import tqdm

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
        tokenizer_1 = self.model.tokenizer_1
        tokenizer_2 = self.model.tokenizer_2
        text_encoder_1 = self.model.text_encoder_1
        text_encoder_2 = self.model.text_encoder_2
        noise_scheduler = self.pipeline.scheduler
        image_processor = self.pipeline.image_processor
        unet = self.pipeline.unet
        vae = self.pipeline.vae
        vae_scale_factor = self.pipeline.vae_scale_factor

        height, width = resolution

        # prepare prompt
        tokenizer_1_output = tokenizer_1(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=tokenizer_1.model_max_length,
            return_tensors="pt",
        )
        tokens_1 = tokenizer_1_output.input_ids.to(self.train_device)
        if hasattr(text_encoder_1.config, "use_attention_mask") and text_encoder_1.config.use_attention_mask:
            tokens_1_attention_mask = tokenizer_1_output.attention_mask.to(self.train_device)
        else:
            tokens_1_attention_mask = None

        tokenizer_2_output = tokenizer_2(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        )
        tokens_2 = tokenizer_2_output.input_ids.to(self.train_device)
        if hasattr(text_encoder_2.config, "use_attention_mask") and text_encoder_2.config.use_attention_mask:
            tokens_2_attention_mask = tokenizer_2_output.attention_mask.to(self.train_device)
        else:
            tokens_2_attention_mask = None

        negative_tokenizer_1_output = tokenizer_1(
            "",
            padding='max_length',
            truncation=True,
            max_length=tokenizer_1.model_max_length,
            return_tensors="pt",
        )
        negative_tokens_1 = negative_tokenizer_1_output.input_ids.to(self.train_device)
        if hasattr(text_encoder_1.config, "use_attention_mask") and text_encoder_1.config.use_attention_mask:
            negative_tokens_1_attention_mask = negative_tokenizer_1_output.attention_mask.to(self.train_device)
        else:
            negative_tokens_1_attention_mask = None

        negative_tokenizer_2_output = tokenizer_2(
            "",
            padding='max_length',
            truncation=True,
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        )
        negative_tokens_2 = negative_tokenizer_2_output.input_ids.to(self.train_device)
        if hasattr(text_encoder_2.config, "use_attention_mask") and text_encoder_2.config.use_attention_mask:
            negative_tokens_2_attention_mask = negative_tokenizer_2_output.attention_mask.to(self.train_device)
        else:
            negative_tokens_2_attention_mask = None

        with torch.autocast(self.train_device.type):
            # TODO: support clip skip
            text_encoder_1_output = text_encoder_1(
                tokens_1,
                attention_mask=tokens_1_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            text_encoder_1_output = text_encoder_1_output.hidden_states[-2]

            text_encoder_2_output = text_encoder_2(
                tokens_2,
                attention_mask=tokens_2_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pooled_text_encoder_2_output = text_encoder_2_output.text_embeds
            text_encoder_2_output = text_encoder_2_output.hidden_states[-2]

            prompt_embedding = torch.concat(
                [text_encoder_1_output, text_encoder_2_output], dim=-1
            )

            negative_text_encoder_1_output = text_encoder_1(
                negative_tokens_1,
                attention_mask=negative_tokens_1_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            negative_text_encoder_1_output = negative_text_encoder_1_output.hidden_states[-2]

            negative_text_encoder_2_output = text_encoder_2(
                negative_tokens_2,
                attention_mask=negative_tokens_2_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            negative_pooled_text_encoder_2_output = negative_text_encoder_2_output.text_embeds
            negative_text_encoder_2_output = negative_text_encoder_2_output.hidden_states[-2]

            negative_prompt_embedding = torch.concat(
                [negative_text_encoder_1_output, negative_text_encoder_2_output], dim=-1
            )

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])

        # prepare timesteps
        noise_scheduler.set_timesteps(steps, device=self.train_device)
        timesteps = noise_scheduler.timesteps

        if force_last_timestep:
            last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                            * (noise_scheduler.config.num_train_timesteps - 1)

            # add the final timestep to force predicting with zero snr
            timesteps = torch.cat([last_timestep, timesteps])

        original_height = height
        original_width = width
        crops_coords_top = 0
        crops_coords_left = 0
        target_height = height
        target_width = width

        add_time_ids = torch.tensor([
            original_height,
            original_width,
            crops_coords_top,
            crops_coords_left,
            target_height,
            target_width
        ]).unsqueeze(dim=0)

        add_time_ids = add_time_ids.to(
            dtype=unet.dtype,
            device=self.train_device,
        )

        # prepare latent image
        num_channels_latents = unet.config.in_channels
        latent_image = torch.randn(
            size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=unet.dtype
        )

        added_cond_kwargs = {
            "text_embeds": torch.concat([pooled_text_encoder_2_output, negative_pooled_text_encoder_2_output], dim=0),
            "time_ids": torch.concat([add_time_ids] * 2, dim=0),
        }

        # denoising loop
        for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
            latent_model_input = torch.cat([latent_image] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            with torch.autocast(self.train_device.type):
                noise_pred = unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=combined_prompt_embedding,
                    added_cond_kwargs=added_cond_kwargs,
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
                noise_pred, timestep, latent_image, return_dict=False
            )[0]

        latent_image = latent_image.to(dtype=vae.dtype)
        image = vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

        return image[0]

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
        if len(self.model.embeddings) > 0:
            tokens = [f"<embedding_{i}>" for i in range(self.model.embeddings[0].token_count)]
            embedding_string = ''.join(tokens)
            prompt = prompt.replace("<embedding>", embedding_string)

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
