import copy
import inspect
from collections.abc import Callable

from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.staged_pipeline import run_staged_pipeline
from modules.util.tqdm_util import tqdm

import torch


@factory.register(BaseModelSampler, ModelType.HUNYUAN_VIDEO)
class HunyuanVideoSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: HunyuanVideoModel,
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
        prompt_embedding, pooled_prompt_embedding, prompt_attention_mask = self.model.encode_text(
            text=sample_config.prompt,
            train_device=self.train_device,
            text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
            text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
        )

        return {
            "prompt_embedding": prompt_embedding,
            "pooled_prompt_embedding": pooled_prompt_embedding,
            "prompt_attention_mask": prompt_attention_mask,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            prompt_embedding: torch.Tensor,
            pooled_prompt_embedding: torch.Tensor,
            prompt_attention_mask: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        self.model.materialize_only("transformer")
        transformer = self.pipeline.transformer
        vae_temporal_scale_factor = 4
        vae_spacial_scale_factor = 8
        num_latent_channels = 16
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)
        num_frames = self.quantize_resolution(sample_config.frames - 1, 4) + 1
        cfg_scale = sample_config.cfg_scale
        diffusion_steps = sample_config.diffusion_steps

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)

        # prepare latent image
        num_latent_frames = (num_frames - 1) // vae_temporal_scale_factor + 1
        latent_image = torch.randn(
            size=(
                1, # batch size
                num_latent_channels,
                num_latent_frames,
                height // vae_spacial_scale_factor,
                width // vae_spacial_scale_factor
            ),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )

        # prepare timesteps
        noise_scheduler.set_timesteps(
            num_inference_steps=diffusion_steps,
            device=self.train_device,
        )
        timesteps = noise_scheduler.timesteps

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            latent_model_input = torch.cat([latent_image])
            expanded_timestep = timestep.expand(latent_model_input.shape[0])

            # handle guidance
            if transformer.config.guidance_embeds:
                guidance = torch.tensor([cfg_scale * 1000.0], device=self.train_device)
                guidance = guidance.expand(latent_model_input.shape[0])
            else:
                guidance = None

            with self.model.transformer_autocast_context:
                # predict the noise residual
                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                    timestep=expanded_timestep,
                    guidance=guidance.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                    pooled_projections=pooled_prompt_embedding.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                    encoder_hidden_states=prompt_embedding.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                    encoder_attention_mask=prompt_attention_mask.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                    return_dict=True
                ).sample

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
        video_processor = self.pipeline.video_processor
        vae = self.pipeline.vae

        latents = latent_image / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]

        image = video_processor.postprocess(image, output_type='pt')

        # postprocess keeps channels ahead of frames, so swap them to [B, F, C, H, W]
        return self.build_video_sampler_output(image.permute(0, 2, 1, 3, 4))

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
                fps=self.model.NATIVE_FPS,
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
