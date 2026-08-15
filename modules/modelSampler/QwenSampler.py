import copy
import inspect
import math
from collections.abc import Callable

from modules.model.QwenModel import QwenModel
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


@factory.register(BaseModelSampler, ModelType.QWEN)
class QwenSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: QwenModel,
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
        cfg_scale = sample_config.cfg_scale

        #unlike other models, Qwen benefits from CFG but is still quite good at CFG 1. Optimize for that:
        batch_size = 2 if cfg_scale > 1.0 else 1
        combined_prompt_embedding, text_attention_mask = self.model.encode_text(
            text=[sample_config.prompt, sample_config.negative_prompt] if cfg_scale > 1.0 else sample_config.prompt,
            batch_size=batch_size,
            train_device=self.train_device,
        )

        return {
            "batch_size": batch_size,
            "combined_prompt_embedding": combined_prompt_embedding,
            "text_attention_mask": text_attention_mask,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            batch_size: int,
            combined_prompt_embedding: torch.Tensor,
            text_attention_mask: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        transformer = self.pipeline.transformer
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

        shift = sample_config.override_shift \
            or self.model.calculate_timestep_shift(latent_image.shape[-2], latent_image.shape[-1])
        latent_image = self.model.pack_latents(latent_image)

        noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=math.log(shift))
        timesteps = noise_scheduler.timesteps

        # denoising loop
        extra_step_kwargs = {}
        #TODO always True for FlowMatchEulerDiscreteScheduler - remove and pass directly?
        #If so, also remove for other models
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator #TODO purpose?

        #FIXME list of lists is not according to type hint, but according to diffusers code
        #https://github.com/huggingface/diffusers/issues/12295
        img_shapes = [[(
            1, #frame for future video model - not batch size
            height // vae_scale_factor // 2,
            width // vae_scale_factor // 2)
        ]] * batch_size

        if torch.all(text_attention_mask):
            text_attention_mask = None

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            latent_model_input = torch.cat([latent_image] * batch_size)
            expanded_timestep = timestep.expand(batch_size)
            noise_pred = transformer(
                hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                timestep=expanded_timestep / 1000,
                encoder_hidden_states=combined_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                encoder_hidden_states_mask=text_attention_mask,
                img_shapes=img_shapes,
                return_dict=True,
            ).sample

            if cfg_scale > 1.0:
                noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            # compute the previous noisy sample x_t -> x_t-1
            latent_image = noise_scheduler.step(
                noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
            )[0]

            on_update_progress(i + 1, len(timesteps))

        latent_image = self.model.unpack_latents(
            latent_image,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        return {
            "latent_image": latent_image,
        }

    @torch.no_grad()
    def __decode(
            self,
            latent_image: torch.Tensor,
    ) -> ModelSamplerOutput:
        self.model.materialize_only("vae")
        image_processor = self.pipeline.image_processor
        vae = self.pipeline.vae

        latents = self.model.unscale_latents(latent_image)
        image = vae.decode(latents, return_dict=False)[0].squeeze(-3)

        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

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
