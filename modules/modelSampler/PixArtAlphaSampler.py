import inspect
from collections.abc import Callable

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import create, factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from tqdm import tqdm


class PixArtAlphaSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: PixArtAlphaModel,
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
            text_encoder_layer_skip: int = 0,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = create.create_noise_scheduler(noise_scheduler, self.pipeline.scheduler, diffusion_steps)
            image_processor = self.pipeline.image_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            vae_scale_factor = self.pipeline.vae_scale_factor

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, tokens_attention_mask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            negative_prompt_embedding, negative_tokens_attention_mask = self.model.encode_text(
                text=negative_prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])
            combined_prompt_attention_mask = torch.cat([negative_tokens_attention_mask, tokens_attention_mask])

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            # prepare latent image
            num_channels_latents = transformer.config.in_channels
            latent_image = torch.randn(
                size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32
            ) * noise_scheduler.init_noise_sigma

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            batch_size = latent_image.shape[0] * 2
            height = latent_image.shape[2] * 8
            width = latent_image.shape[3] * 8
            resolution = torch.tensor([height, width]).repeat(batch_size, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
            resolution = resolution \
                .to(dtype=self.model.train_dtype.torch_dtype(), device=self.train_device)
            aspect_ratio = aspect_ratio \
                .to(dtype=self.model.train_dtype.torch_dtype(), device=self.train_device)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            # denoising loop
            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

                # predict the noise residual
                noise_pred = transformer(
                    latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    encoder_hidden_states=combined_prompt_embedding.to(
                        dtype=self.model.train_dtype.torch_dtype()),
                    encoder_attention_mask=combined_prompt_attention_mask.to(
                        dtype=self.model.train_dtype.torch_dtype()),
                    timestep=timestep.expand(latent_model_input.shape[0]),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # extract mean
                noise_pred = noise_pred.chunk(2, dim=1)[0]

                # cfg
                noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # decode
            self.model.vae_to(self.train_device)

            latent_image = latent_image.to(dtype=vae.dtype)
            image = vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]

            do_denormalize = [True] * image.shape[0]
            image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

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
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, PixArtAlphaSampler, ModelType.PIXART_ALPHA)
factory.register(BaseModelSampler, PixArtAlphaSampler, ModelType.PIXART_SIGMA)
