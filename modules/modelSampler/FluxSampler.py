import copy
import inspect
import os
from pathlib import Path
from typing import Callable

from modules.model.FluxModel import FluxModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.torch_util import torch_gc

import torch

from PIL import Image
from tqdm import tqdm


class FluxSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: FluxModel,
            model_type: ModelType,
    ):
        super(FluxSampler, self).__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    def __calculate_shift(
            self,
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            cfg_rescale: float = 0.7,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_3_layer_skip: int = 0,
            force_last_timestep: bool = False,
            prior_attention_mask: bool = False,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> Image.Image:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            image_processor = self.pipeline.image_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            vae_scale_factor = self.pipeline.vae_scale_factor // 2

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, pooled_prompt_embedding = self.model.encode_text(
                text = prompt,
                train_device = self.train_device,
                batch_size=1,
                text_encoder_1_layer_skip = text_encoder_1_layer_skip,
                text_encoder_2_layer_skip = text_encoder_2_layer_skip,
                apply_attention_mask = prior_attention_mask,
            )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent image
            num_channels_latents = transformer.config.in_channels // 4
            latent_image = torch.randn(
                size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            image_ids = self.model.prepare_latent_image_ids(
                latent_image.shape[0],
                height // vae_scale_factor,
                width // vae_scale_factor,
                self.train_device,
                self.model.train_dtype.torch_dtype()
            )

            latent_image = self.model.pack_latents(
                latent_image,
                latent_image.shape[0],
                latent_image.shape[1],
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            image_seq_len = latent_image.shape[1]

            # prepare timesteps
            mu = self.__calculate_shift(
                image_seq_len,
                noise_scheduler.config.base_image_seq_len,
                noise_scheduler.config.max_image_seq_len,
                noise_scheduler.config.base_shift,
                noise_scheduler.config.max_shift,
            )
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=mu)
            timesteps = noise_scheduler.timesteps

            if force_last_timestep:
                last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                                * (noise_scheduler.config.num_train_timesteps - 1)

                # add the final timestep to force predicting with zero snr
                timesteps = torch.cat([last_timestep, timesteps])

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            text_ids = torch.zeros(latent_image.shape[0], prompt_embedding.shape[1], 3, device=self.train_device)

            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image])
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                # handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([cfg_scale], device=self.train_device)
                    guidance = guidance.expand(latent_model_input.shape[0])
                else:
                    guidance = None

                # predict the noise residual
                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=expanded_timestep / 1000,
                    guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),
                    pooled_projections=pooled_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    encoder_hidden_states=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    txt_ids=text_ids.to(dtype=self.model.train_dtype.torch_dtype()),
                    img_ids=image_ids.to(dtype=self.model.train_dtype.torch_dtype()),
                    joint_attention_kwargs=None,
                    return_dict=True
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            latent_image = self.model.unpack_latents(
                latent_image,
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            # decode
            self.model.vae_to(self.train_device)

            latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]

            do_denormalize = [True] * image.shape[0]
            image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

            self.model.vae_to(self.temp_device)

            return image[0]

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            on_sample: Callable[[Image], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        prompt = self.model.add_embeddings_to_prompt(sample_config.prompt)
        negative_prompt = self.model.add_embeddings_to_prompt(sample_config.negative_prompt)

        image = self.__sample_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=sample_config.height,
            width=sample_config.width,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            cfg_rescale=0.7 if sample_config.force_last_timestep else 0.0,
            text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
            text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
            text_encoder_3_layer_skip=sample_config.text_encoder_3_layer_skip,
            force_last_timestep=sample_config.force_last_timestep,
            prior_attention_mask=sample_config.prior_attention_mask,
            on_update_progress=on_update_progress,
        )

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination, format=image_format.pil_format())

        on_sample(image)
