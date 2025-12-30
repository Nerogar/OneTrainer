import copy
import inspect
from collections.abc import Callable

from modules.model.HunyuanVideoModel import HunyuanVideoModel
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

from PIL import Image
from tqdm import tqdm


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
        self.pipeline = model.create_pipeline(use_original_modules=False)

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            num_frames: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            transformer_attention_mask: bool = False,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            video_processor = self.pipeline.video_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            vae_temporal_scale_factor = 4
            vae_spacial_scale_factor = 8
            num_latent_channels = 16

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, pooled_prompt_embedding, prompt_attention_mask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=text_encoder_2_layer_skip,
            )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

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

            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
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

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # decode
            self.model.vae_to(self.train_device)

            latents = latent_image / vae.config.scaling_factor
            image = vae.decode(latents, return_dict=False)[0]

            image = video_processor.postprocess(image, output_type='pt')

            self.model.vae_to(self.temp_device)
            torch_gc()

            is_image = image.shape[2] == 1
            if is_image:
                image = image.view((image.shape[0], image.shape[1], image.shape[3], image.shape[4]))
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")
                image = Image.fromarray(image[0])

                return ModelSamplerOutput(
                    file_type=FileType.IMAGE,
                    data=image,
                )
            else:
                image = image.cpu().permute(0, 2, 3, 4, 1).float()
                image = (image.clamp(0, 1) * 255).round().to(dtype=torch.int8)
                image = image[0]

                return ModelSamplerOutput(
                    file_type=FileType.VIDEO,
                    data=image,
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
            num_frames=self.quantize_resolution(sample_config.frames - 1, 4) + 1,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
            text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
            transformer_attention_mask=sample_config.transformer_attention_mask,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, HunyuanVideoSampler, ModelType.HUNYUAN_VIDEO)
