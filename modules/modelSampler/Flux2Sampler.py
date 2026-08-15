import copy
import inspect
import math
from collections.abc import Callable

from modules.model.Flux2Model import Flux2Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.staged_pipeline import run_staged_pipeline
from modules.util.tqdm_util import tqdm

import torch

from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

import numpy as np

VAE_SCALE_FACTOR = 8
NUM_LATENT_CHANNELS = 32
PATCH_SIZE = 2


@factory.register(BaseModelSampler, ModelType.FLUX_2)
class Flux2Sampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: Flux2Model,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        self.model.materialize_only("text_encoder")
        transformer = self.pipeline.transformer

        batch_size = 2 if sample_config.cfg_scale > 1.0 and not transformer.config.guidance_embeds else 1
        prompt_embedding = self.model.encode_text(
            text=[sample_config.prompt, sample_config.negative_prompt] if batch_size == 2 else sample_config.prompt,
            train_device=self.train_device,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
        )

        return {
            "batch_size": batch_size,
            "prompt_embedding": prompt_embedding,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            batch_size: int,
            prompt_embedding: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        transformer = self.pipeline.transformer
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)
        diffusion_steps = sample_config.diffusion_steps

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        text_ids = self.model.prepare_text_ids(prompt_embedding)

        # prepare latent image
        latent_image = torch.randn(
            size=(1, NUM_LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )

        latent_image = self.model.patchify_latents(latent_image)
        image_ids = self.model.prepare_latent_image_ids(latent_image)

        latent_image = self.model.pack_latents(latent_image)
        image_seq_len = latent_image.shape[1]
        # the override is a shift factor, the same quantity the other flow-matching samplers pass as log(shift)
        mu = math.log(sample_config.override_shift) if sample_config.override_shift \
            else compute_empirical_mu(image_seq_len, diffusion_steps)

        # prepare timesteps
        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
        #TODO for other models, too? This is different than with sigmas=None
        sigmas = np.linspace(1.0, 1 / diffusion_steps, diffusion_steps)
        noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=mu, sigmas=sigmas)
        timesteps = noise_scheduler.timesteps

        extra_step_kwargs = {} #TODO remove
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        guidance = (torch.tensor([sample_config.cfg_scale], device=self.train_device, dtype=self.model.train_dtype.torch_dtype())
                    if transformer.config.guidance_embeds else None)
        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            latent_model_input = torch.cat([latent_image] * batch_size)
            expanded_timestep = timestep.expand(latent_model_input.shape[0])

            noise_pred = transformer(
                hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                timestep=expanded_timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                txt_ids=text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=True
            ).sample

            if batch_size == 2:
                noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + sample_config.cfg_scale * (noise_pred_positive - noise_pred_negative)

            latent_image = noise_scheduler.step(noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs)[0]

            on_update_progress(i + 1, len(timesteps))

        return {
            "latent_image": latent_image,
            "height": height,
            "width": width,
        }

    @torch.no_grad()
    def __decode(
            self,
            height: int,
            width: int,
            latent_image: torch.Tensor,
    ) -> ModelSamplerOutput:
        self.model.materialize_only("vae")
        image_processor = self.pipeline.image_processor
        vae = self.pipeline.vae

        latent_image = self.model.unpack_latents(
            latent_image,
            height // VAE_SCALE_FACTOR // PATCH_SIZE,
            width // VAE_SCALE_FACTOR // PATCH_SIZE,
        )
        latents = self.model.unscale_latents(latent_image)
        latents = self.model.unpatchify_latents(latents)

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type='pil')

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
    ) -> list[ModelSamplerOutput]:
        batch_progress = self.batch_progress_callback(sample_configs, on_update_progress)

        with self.model.autocast_context:
            sampler_outputs = run_staged_pipeline(
                [("encoding", self.__encode), ("denoising", self.__denoise), ("decoding", self.__decode)],
                {"sample_config": sample_configs},
                {"on_update_progress": batch_progress},
            )

        for sampler_output, destination in zip(sampler_outputs, destinations, strict=True):
            self.save_sampler_output(
                sampler_output, destination,
                image_format, video_format, audio_format,
            )

        return sampler_outputs

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

        on_sample(sampler_output)
