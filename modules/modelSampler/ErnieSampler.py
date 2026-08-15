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
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.staged_pipeline import run_staged_pipeline
from modules.util.tqdm_util import tqdm

import torch

import numpy as np
from PIL import Image as PILImage


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
    def __encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        self.model.materialize_only("text_encoder")
        batch_size = 2 if sample_config.cfg_scale > 1.0 else 1
        text_bth, text_lens = self.model.encode_text(
            train_device=self.train_device,
            text=[sample_config.prompt, sample_config.negative_prompt] if batch_size == 2 else sample_config.prompt,
        )

        return {
            "batch_size": batch_size,
            "text_bth": text_bth,
            "text_lens": text_lens,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            batch_size: int,
            text_bth: torch.Tensor,
            text_lens: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        transformer = self.pipeline.transformer
        vae_scale_factor = 8
        num_latent_channels = 32
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)
        cfg_scale = sample_config.cfg_scale
        diffusion_steps = sample_config.diffusion_steps
        dtype = self.model.train_dtype.torch_dtype()

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)

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

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
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

        return {
            "latent_image": latent_image,
        }

    @torch.no_grad()
    def __decode(
            self,
            latent_image: torch.Tensor,
    ) -> ModelSamplerOutput:
        self.model.materialize_only("vae")
        vae = self.pipeline.vae

        # unscale and unpatchify
        latents = self.model.unscale_latents(latent_image)
        latents = self.model.unpatchify_latents(latents)

        image = vae.decode(latents, return_dict=False)[0]
        # no VaeImageProcessor — pipeline does this manually
        image = (image.clamp(-1, 1) + 1) / 2
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = [PILImage.fromarray((img * 255).astype(np.uint8)) for img in image]

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
