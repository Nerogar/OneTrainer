import copy
import inspect
from collections.abc import Callable

from modules.model.AnimaModel import AnimaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.staged_pipeline import move_tensors
from modules.util.tqdm_util import tqdm

import torch

from diffusers import VaeImageProcessor

import numpy as np


@factory.register(BaseModelSampler, ModelType.ANIMA)
class AnimaSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: AnimaModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    @torch.no_grad()
    def __encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        self.model.materialize_only("text_encoder")
        batch_size = 2 if sample_config.cfg_scale > 1.0 else 1
        combined_prompt_embedding = self.model.encode_text(
            text=[sample_config.prompt, sample_config.negative_prompt] if sample_config.cfg_scale > 1.0 else sample_config.prompt,
            batch_size=batch_size,
            train_device=self.train_device,
        )

        return {
            "batch_size": batch_size,
            "combined_prompt_embedding": combined_prompt_embedding,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            batch_size: int,
            combined_prompt_embedding: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        combined_prompt_embedding = move_tensors(combined_prompt_embedding, self.train_device)
        transformer = self.model.transformer
        vae_scale_factor = 8
        num_latent_channels = 16
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)
        cfg_scale = sample_config.cfg_scale
        diffusion_steps = sample_config.diffusion_steps

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)

        # prepare latent image
        latent_image = torch.randn(
            size=(1, num_latent_channels, 1, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )

        sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
        noise_scheduler.set_timesteps(sigmas=sigmas, device=self.train_device)
        timesteps = noise_scheduler.timesteps

        padding_mask = latent_image.new_zeros(
            1, 1, height, width, dtype=transformer.dtype,
        )

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            latent_model_input = torch.cat([latent_image] * batch_size)
            expanded_timestep = timestep.expand(batch_size) / noise_scheduler.config.num_train_timesteps
            noise_pred = transformer(
                hidden_states=latent_model_input.to(dtype=transformer.dtype),
                timestep=expanded_timestep,
                encoder_hidden_states=combined_prompt_embedding.to(dtype=transformer.dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            if cfg_scale > 1.0:
                noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            latent_image = noise_scheduler.step(
                noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
            )[0]

            on_update_progress(i + 1, len(timesteps))

        return {
            "latent_image": latent_image,
        }

    @torch.no_grad()
    def __decode(
            self,
            latent_image: torch.Tensor,
    ) -> ModelSamplerOutput:
        self.model.materialize_only("vae")
        latent_image = move_tensors(latent_image, self.train_device)
        vae = self.model.vae

        latents = self.model.unscale_latents(latent_image)
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

        return ModelSamplerOutput(
            file_type=FileType.IMAGE,
            data=image[0],
        )

    def sample_all(
            self,
            sample_configs: list[SampleConfig],
            destinations: list[str],
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
            *,
            on_sample: Callable[[int, ModelSamplerOutput], None] | None = None,
            on_error: Callable[[int, str, Exception], None] | None = None,
            offload_intermediates: bool = True,
    ) -> list[ModelSamplerOutput | None]:
        with self.model.autocast_context:
            return self._sample_staged(
                [("encoding", self.__encode), ("denoising", self.__denoise), ("decoding", self.__decode)],
                sample_configs, destinations, image_format, video_format, audio_format,
                on_update_progress, on_sample, on_error, offload_intermediates,
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # single-sample entry point: a staged batch of one
        sampler_output = self.sample_all(
            [sample_config], [destination],
            image_format, video_format, audio_format,
            on_update_progress=on_update_progress,
        )[0]

        assert sampler_output is not None  # single-sample calls propagate errors
        on_sample(sampler_output)
