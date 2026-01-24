import copy
import inspect
import math
from collections.abc import Callable

from modules.model.FluxModel import FluxModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.image_util import load_image
from modules.util.torch_util import torch_gc

import torch
from torch import nn
from torchvision.transforms import transforms

from tqdm import tqdm


class FluxSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: FluxModel,
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
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_2_sequence_length: int | None = None,
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
            image_processor = self.pipeline.image_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            vae_scale_factor = 8
            num_latent_channels = 16

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, pooled_prompt_embedding = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=text_encoder_2_layer_skip,
                text_encoder_2_sequence_length=text_encoder_2_sequence_length,
                apply_attention_mask=transformer_attention_mask,
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

            image_ids = self.model.prepare_latent_image_ids(
                height // vae_scale_factor,
                width // vae_scale_factor,
                self.train_device,
                self.model.train_dtype.torch_dtype()
            )

            shift = self.model.calculate_timestep_shift(latent_image.shape[-2], latent_image.shape[-1])
            latent_image = self.model.pack_latents(latent_image)

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=math.log(shift))
            timesteps = noise_scheduler.timesteps

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            text_ids = torch.zeros(prompt_embedding.shape[1], 3, device=self.train_device)

            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image])
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                # handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([cfg_scale], device=self.train_device)
                    guidance = guidance.expand(latent_model_input.shape[0])
                else:
                    guidance = None

                # predict the noise residual
                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=expanded_timestep / 1000,
                    guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),
                    pooled_projections=pooled_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    encoder_hidden_states=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=True
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()
            latent_image = self.model.unpack_latents(
                latent_image,
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            # decode
            self.model.vae_to(self.train_device)

            latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]

            do_denormalize = [True] * image.shape[0] #TODO remove and test, from Flux and other models. True is the default
            image = image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
            )

    def __create_erode_kernel(self, device, dtype=torch.float32):
        kernel_radius = 2

        kernel_size = kernel_radius * 2 + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size, dtype=dtype) / (kernel_size * kernel_size)
        kernel = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding_mode='replicate',
            padding=kernel_radius
        ).to(dtype)
        kernel.weight.data = kernel_weights
        kernel.requires_grad_(False)
        kernel.to(device)
        return kernel

    @torch.no_grad()
    def __sample_inpainting(
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
            sample_inpainting: bool = False,
            base_image_path: str = "",
            mask_image_path: str = "",
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_2_sequence_length: int | None = None,
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
            image_processor = self.pipeline.image_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            vae_scale_factor = 8
            num_latent_channels = 16

            # prepare conditioning image
            self.model.vae_to(self.train_device)

            if sample_inpainting:
                t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(
                        (height, width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                    ),
                ])

                image = load_image(base_image_path, convert_mode="RGB")
                image = t(image).to(
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )

                mask = load_image(mask_image_path, convert_mode='L')
                mask = t(mask).to(
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )

                erode_kernel = self.__create_erode_kernel(self.train_device, dtype=self.model.train_dtype.torch_dtype())
                eroded_mask = erode_kernel(mask)
                eroded_mask = (eroded_mask > 0.5).to(dtype=self.model.train_dtype.torch_dtype())

                image = (image * 2.0) - 1.0
                conditioning_image = (image * (1 - eroded_mask))
                conditioning_image = conditioning_image.unsqueeze(0)

                latent_conditioning_image = vae.encode(conditioning_image).latent_dist.mode()
                latent_conditioning_image = (latent_conditioning_image - vae.config.shift_factor) \
                                            * vae.config.scaling_factor

                latent_conditioning_image = self.model.pack_latents(latent_conditioning_image)

                # batch_size, height, 8, width, 8
                mask = mask.view(
                    mask.shape[0],
                    height // vae_scale_factor,
                    vae_scale_factor,
                    width // vae_scale_factor,
                    vae_scale_factor,
                )
                # batch_size, 8, 8, height, width
                mask = mask.permute(0, 2, 4, 1, 3)
                # batch_size, 8*8, height, width
                mask = mask.reshape(
                    mask.shape[0],
                    vae_scale_factor * vae_scale_factor,
                    height // vae_scale_factor,
                    width // vae_scale_factor,
                )

                latent_mask = self.model.pack_latents(mask)
            else:
                conditioning_image = torch.zeros(
                    (1, 3, height, width),
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )
                latent_conditioning_image = vae.encode(conditioning_image).latent_dist.mode()
                latent_conditioning_image = (latent_conditioning_image - vae.config.shift_factor) \
                                            * vae.config.scaling_factor

                latent_conditioning_image = self.model.pack_latents(latent_conditioning_image)

                latent_mask = torch.ones(
                    size=(1, (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2), 256),
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device
                )

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, pooled_prompt_embedding = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=text_encoder_2_layer_skip,
                text_encoder_2_sequence_length=text_encoder_2_sequence_length,
                apply_attention_mask=transformer_attention_mask,
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

            image_ids = self.model.prepare_latent_image_ids(
                height // vae_scale_factor,
                width // vae_scale_factor,
                self.train_device,
                self.model.train_dtype.torch_dtype()
            )

            shift = self.model.calculate_timestep_shift(latent_image.shape[-2], latent_image.shape[-1])
            latent_image = self.model.pack_latents(latent_image)
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=math.log(shift))
            timesteps = noise_scheduler.timesteps

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            text_ids = torch.zeros(prompt_embedding.shape[1], 3, device=self.train_device)

            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image])
                latent_model_input = torch.concat(
                    [latent_model_input, latent_conditioning_image, latent_mask], -1
                )
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                # handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([cfg_scale], device=self.train_device)
                    guidance = guidance.expand(latent_model_input.shape[0])
                else:
                    guidance = None

                # predict the noise residual
                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=expanded_timestep / 1000,
                    guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),
                    pooled_projections=pooled_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    encoder_hidden_states=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    txt_ids=text_ids.to(dtype=self.model.train_dtype.torch_dtype()),
                    img_ids=image_ids.to(dtype=self.model.train_dtype.torch_dtype()),
                    joint_attention_kwargs=None,
                    return_dict=True
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            latent_image = self.model.unpack_latents(
                latent_image,
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            # decode
            self.model.vae_to(self.train_device)

            latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]

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
        if self.model_type.has_conditioning_image_input():
            sampler_output = self.__sample_inpainting(
                prompt=sample_config.prompt,
                negative_prompt=sample_config.negative_prompt,
                height=self.quantize_resolution(sample_config.height, 64),
                width=self.quantize_resolution(sample_config.width, 64),
                seed=sample_config.seed,
                random_seed=sample_config.random_seed,
                diffusion_steps=sample_config.diffusion_steps,
                cfg_scale=sample_config.cfg_scale,
                noise_scheduler=sample_config.noise_scheduler,
                sample_inpainting=sample_config.sample_inpainting,
                base_image_path=sample_config.base_image_path,
                mask_image_path=sample_config.mask_image_path,
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
                text_encoder_2_sequence_length=sample_config.text_encoder_2_sequence_length,
                transformer_attention_mask=sample_config.transformer_attention_mask,
                on_update_progress=on_update_progress,
            )
        else:
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
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
                text_encoder_2_sequence_length=sample_config.text_encoder_2_sequence_length,
                transformer_attention_mask=sample_config.transformer_attention_mask,
                on_update_progress=on_update_progress,
            )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, FluxSampler, ModelType.FLUX_DEV_1)
factory.register(BaseModelSampler, FluxSampler, ModelType.FLUX_FILL_DEV_1)
