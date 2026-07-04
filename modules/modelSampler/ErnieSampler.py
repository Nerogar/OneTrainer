import copy
from collections.abc import Callable

from modules.model.ErnieModel import ErnieModel
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

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


@factory.register(BaseModelSampler, ModelType.ERNIE)
class ErnieSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: ErnieModel,
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
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            vae = self.pipeline.vae

            vae_scale_factor = 8
            num_latent_channels = 32

            # encode text
            self.model.text_encoder_to(self.train_device)

            batch_size = 2 if cfg_scale > 1.0 else 1
            text_bth, text_lens = self.model.encode_text(
                train_device=self.train_device,
                text=[prompt, negative_prompt] if batch_size == 2 else prompt,
            )
            dtype = self.model.train_dtype.torch_dtype()

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latents
            latent_image = torch.randn(
                size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )
            latent_image = self.model.patchify_latents(latent_image)

            sigmas = np.linspace(1.0, 1 / diffusion_steps, diffusion_steps)
            noise_scheduler.set_timesteps(sigmas=sigmas, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            self.model.transformer_to(self.train_device)
            transformer = self.pipeline.transformer

            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image] * batch_size)
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=dtype),
                    timestep=expanded_timestep,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                if batch_size == 2:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                latent_image = noise_scheduler.step(noise_pred, timestep, latent_image,
                                                    return_dict=False)[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()
            self.model.vae_to(self.train_device)

            # unscale and unpatchify
            latents = self.model.unscale_latents(latent_image)
            latents = self.model.unpatchify_latents(latents)

            image = vae.decode(latents, return_dict=False)[0]
            # no VaeImageProcessor — pipeline does this manually
            image = (image.clamp(-1, 1) + 1) / 2
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = [PILImage.fromarray((img * 255).astype(np.uint8)) for img in image]

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
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
