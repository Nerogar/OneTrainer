import copy
import inspect
from collections.abc import Callable

from modules.model.Flux2Model import Flux2Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

import numpy as np
from tqdm import tqdm


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
            text_encoder_sequence_length: int | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
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

            vae_scale_factor = 8
            num_latent_channels = 32
            patch_size = 2

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            batch_size = 2 if cfg_scale > 1.0 and not transformer.config.guidance_embeds else 1
            prompt_embedding = self.model.encode_text(
                text=[prompt, negative_prompt] if batch_size == 2 else prompt,
                train_device=self.train_device,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent image
            latent_image = torch.randn(
                size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            latent_image = self.model.patchify_latents(latent_image)
            image_ids = self.model.prepare_latent_image_ids(latent_image)

            latent_image = self.model.pack_latents(latent_image)
            image_seq_len = latent_image.shape[1]
            mu = compute_empirical_mu(image_seq_len, diffusion_steps)

            # prepare timesteps
            #TODO for other models, too? This is different than with sigmas=None
            sigmas = np.linspace(1.0, 1 / diffusion_steps, diffusion_steps)
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=mu, sigmas=sigmas)
            timesteps = noise_scheduler.timesteps

            # denoising loop
            extra_step_kwargs = {} #TODO remove
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            text_ids = self.model.prepare_text_ids(prompt_embedding)


            self.model.transformer_to(self.train_device)
            guidance = (torch.tensor([cfg_scale], device=self.train_device, dtype=self.model.train_dtype.torch_dtype())
                        if transformer.config.guidance_embeds else None)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
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
                    noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                latent_image = noise_scheduler.step(noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs)[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()
            self.model.vae_to(self.train_device)

            latent_image = self.model.unpack_latents(
                latent_image,
                height // vae_scale_factor // patch_size,
                width // vae_scale_factor // patch_size,
            )
            latents = self.model.unscale_latents(latent_image)
            latents = self.model.unpatchify_latents(latents)

            image = vae.decode(latents, return_dict=False)[0]

            image = image_processor.postprocess(image, output_type='pil')

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
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
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, Flux2Sampler, ModelType.FLUX_2)
