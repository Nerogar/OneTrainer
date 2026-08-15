import copy
import inspect
from collections.abc import Callable

from modules.model.StableDiffusion3Model import StableDiffusion3Model
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


@factory.register(BaseModelSampler, ModelType.STABLE_DIFFUSION_3)
@factory.register(BaseModelSampler, ModelType.STABLE_DIFFUSION_35)
class StableDiffusion3Sampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusion3Model,
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
        self.model.materialize_only_text_encoders()
        prompt_embedding, pooled_prompt_embedding = self.model.combine_text_encoder_output(
            *self.model.encode_text(
                text=sample_config.prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
                text_encoder_3_layer_skip=sample_config.text_encoder_3_layer_skip,
                apply_attention_mask=sample_config.transformer_attention_mask,
            ))

        negative_prompt_embedding, negative_pooled_prompt_embedding = self.model.combine_text_encoder_output(
            *self.model.encode_text(
                text=sample_config.negative_prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
                text_encoder_3_layer_skip=sample_config.text_encoder_3_layer_skip,
                apply_attention_mask=sample_config.transformer_attention_mask,
            ))

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding], dim=0)
        combined_pooled_prompt_embedding = torch.cat(
            [negative_pooled_prompt_embedding, pooled_prompt_embedding], dim=0)

        return {
            "combined_prompt_embedding": combined_prompt_embedding,
            "combined_pooled_prompt_embedding": combined_pooled_prompt_embedding,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            combined_prompt_embedding: torch.Tensor,
            combined_pooled_prompt_embedding: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        transformer = self.pipeline.transformer
        vae_scale_factor = self.pipeline.vae_scale_factor
        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)
        cfg_scale = sample_config.cfg_scale
        diffusion_steps = sample_config.diffusion_steps

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)

        # prepare timesteps
        noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
        timesteps = noise_scheduler.timesteps

        # prepare latent image
        num_channels_latents = transformer.config.in_channels
        latent_image = torch.randn(
            size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            latent_model_input = torch.cat([latent_image] * 2)
            expanded_timestep = timestep.expand(latent_model_input.shape[0])
            # Don't seem to scale the latents in SD3.

            # predict the noise residual
            noise_pred = transformer(
                hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                timestep=expanded_timestep,
                encoder_hidden_states=combined_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                pooled_projections=combined_pooled_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                return_dict=True
            ).sample

            # cfg
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            # compute the previous noisy sample x_t -> x_t-1
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
        image_processor = self.pipeline.image_processor
        vae = self.pipeline.vae

        latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]

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
