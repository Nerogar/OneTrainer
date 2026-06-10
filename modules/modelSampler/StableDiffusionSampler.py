import inspect
from collections.abc import Callable

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import create, factory
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


class StableDiffusionSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusionModel,
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
            cfg_rescale: float = 0.7,
            text_encoder_layer_skip: int = 0,
            force_last_timestep: bool = False,
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
            unet = self.pipeline.unet
            vae = self.pipeline.vae
            vae_scale_factor = self.pipeline.vae_scale_factor

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )
            negative_prompt_embedding = self.model.encode_text(
                text=negative_prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding]) \
                .to(dtype=self.model.train_dtype.torch_dtype())

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            if force_last_timestep:
                last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                                * (noise_scheduler.config.num_train_timesteps - 1)

                # add the final timestep to force predicting with zero snr if it's not already here
                if timesteps[0] != last_timestep:
                    noise_scheduler.set_timesteps(diffusion_steps + 1, device=self.train_device)
                    timesteps = torch.cat([last_timestep, timesteps])

            # prepare latent image
            num_channels_latents = unet.config.in_channels
            latent_image = torch.randn(
                size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            ) * noise_scheduler.init_noise_sigma

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            # denoising loop
            self.model.unet_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = torch.cat([latent_image] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=combined_prompt_embedding,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # cfg
                noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                if cfg_rescale > 0.0:
                    # From: Common Diffusion Noise Schedules and Sample Steps are Flawed (https://arxiv.org/abs/2305.08891)
                    std_positive = noise_pred_positive.std(dim=list(range(1, noise_pred_positive.ndim)), keepdim=True)
                    std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                    noise_pred_rescaled = noise_pred * (std_positive / std_pred)
                    noise_pred = (
                            cfg_rescale * noise_pred_rescaled + (1 - cfg_rescale) * noise_pred
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.unet_to(self.temp_device)
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
            cfg_rescale: float = 0.7,
            sample_inpainting: bool = False,
            base_image_path: str = "",
            mask_image_path: str = "",
            text_encoder_layer_skip: int = 0,
            force_last_timestep: bool = False,
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
            unet = self.pipeline.unet
            vae = self.pipeline.vae
            vae_scale_factor = self.pipeline.vae_scale_factor

            # prepare conditioning image
            self.model.vae_to(self.train_device)

            if sample_inpainting:
                t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(
                        (height, width), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True
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

                latent_conditioning_image = vae.encode(
                    conditioning_image).latent_dist.mode() * vae.config.scaling_factor

                rescale_mask = transforms.Resize(
                    (round(mask.shape[1] // 8), round(mask.shape[2] // 8)),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
                latent_mask = rescale_mask(mask)
                latent_mask = (latent_mask > 0).to(dtype=self.model.train_dtype.torch_dtype())
                latent_mask = latent_mask.unsqueeze(0)
            else:
                conditioning_image = torch.zeros(
                    (1, 3, height, width),
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )
                latent_conditioning_image = vae.encode(conditioning_image).latent_dist.mode() * vae.config.scaling_factor
                latent_mask = torch.ones(
                    size=(1, 1, latent_conditioning_image.shape[2], latent_conditioning_image.shape[3]),
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device
                )

            self.model.vae_to(self.temp_device)
            torch_gc()

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )
            negative_prompt_embedding = self.model.encode_text(
                text=negative_prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding]) \
                .to(dtype=self.model.train_dtype.torch_dtype())

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            if force_last_timestep:
                last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                                * (noise_scheduler.config.num_train_timesteps - 1)

                # add the final timestep to force predicting with zero snr if it's not already here
                if timesteps[0] != last_timestep:
                    noise_scheduler.set_timesteps(diffusion_steps + 1, device=self.train_device)
                    timesteps = torch.cat([last_timestep, timesteps])

            # prepare latent image
            num_channels_latents = latent_conditioning_image.shape[1]
            latent_image = torch.randn(
                size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            ) * noise_scheduler.init_noise_sigma

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            # denoising loop
            self.model.unet_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = noise_scheduler.scale_model_input(latent_image, timestep)
                latent_model_input = torch.concat(
                    [latent_model_input, latent_mask, latent_conditioning_image], 1
                )
                latent_model_input = torch.cat([latent_model_input] * 2)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=combined_prompt_embedding,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # cfg
                noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

                if cfg_rescale > 0.0:
                    # From: Common Diffusion Noise Schedules and Sample Steps are Flawed (https://arxiv.org/abs/2305.08891)
                    std_positive = noise_pred_positive.std(dim=list(range(1, noise_pred_positive.ndim)), keepdim=True)
                    std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                    noise_pred_rescaled = noise_pred * (std_positive / std_pred)
                    noise_pred = (
                            cfg_rescale * noise_pred_rescaled + (1 - cfg_rescale) * noise_pred
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.unet_to(self.temp_device)
            torch_gc()

            #decode
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
        if self.model_type.has_conditioning_image_input():
            sampler_output = self.__sample_inpainting(
                prompt=sample_config.prompt,
                negative_prompt=sample_config.negative_prompt,
                height=self.quantize_resolution(sample_config.height, 8),
                width=self.quantize_resolution(sample_config.width, 8),
                seed=sample_config.seed,
                random_seed=sample_config.random_seed,
                diffusion_steps=sample_config.diffusion_steps,
                cfg_scale=sample_config.cfg_scale,
                noise_scheduler=sample_config.noise_scheduler,
                cfg_rescale=0.7 if sample_config.force_last_timestep else 0.0,
                sample_inpainting=sample_config.sample_inpainting,
                base_image_path=sample_config.base_image_path,
                mask_image_path=sample_config.mask_image_path,
                text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
                force_last_timestep=sample_config.force_last_timestep,
                on_update_progress=on_update_progress,
            )
        else:
            sampler_output = self.__sample_base(
                prompt=sample_config.prompt,
                negative_prompt=sample_config.negative_prompt,
                height=self.quantize_resolution(sample_config.height, 8),
                width=self.quantize_resolution(sample_config.width, 8),
                seed=sample_config.seed,
                random_seed=sample_config.random_seed,
                diffusion_steps=sample_config.diffusion_steps,
                cfg_scale=sample_config.cfg_scale,
                noise_scheduler=sample_config.noise_scheduler,
                cfg_rescale=0.7 if sample_config.force_last_timestep else 0.0,
                text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
                force_last_timestep=sample_config.force_last_timestep,
                on_update_progress=on_update_progress,
            )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_15)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_15_INPAINTING)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_20)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_20_BASE)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_20_INPAINTING)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_20_DEPTH)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_21)
factory.register(BaseModelSampler, StableDiffusionSampler, ModelType.STABLE_DIFFUSION_21_BASE)
