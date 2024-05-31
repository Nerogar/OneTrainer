import inspect
import os
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from tqdm import tqdm

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util import create
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.torch_util import torch_gc


class StableDiffusionSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusionModel,
            model_type: ModelType,
    ):
        super(StableDiffusionSampler, self).__init__(train_device, temp_device)

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
    ) -> Image.Image:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            tokenizer = self.pipeline.tokenizer
            text_encoder = self.pipeline.text_encoder
            noise_scheduler = create.create_noise_scheduler(noise_scheduler, self.pipeline.scheduler, diffusion_steps)
            image_processor = self.pipeline.image_processor
            unet = self.pipeline.unet
            vae = self.pipeline.vae
            vae_scale_factor = self.pipeline.vae_scale_factor

            # prepare prompt
            self.model.text_encoder_to(self.train_device)
            tokenizer_output = tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.train_device)
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)
            else:
                tokens_attention_mask = None

            negative_tokenizer_output = tokenizer(
                negative_prompt,
                padding='max_length',
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            negative_tokens = negative_tokenizer_output.input_ids.to(self.train_device)
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                negative_tokens_attention_mask = negative_tokenizer_output.attention_mask.to(self.train_device)
            else:
                negative_tokens_attention_mask = None

            text_encoder_output = text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            final_layer_norm = text_encoder.text_model.final_layer_norm
            prompt_embedding = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
            )

            text_encoder_output = text_encoder(
                negative_tokens,
                attention_mask=negative_tokens_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            final_layer_norm = text_encoder.text_model.final_layer_norm
            negative_prompt_embedding = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
            )

            combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            if force_last_timestep:
                last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                                * (noise_scheduler.config.num_train_timesteps - 1)

                # add the final timestep to force predicting with zero snr
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

            return image[0]

    def __create_erode_kernel(self, device):
        kernel_radius = 2

        kernel_size = kernel_radius * 2 + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding_mode='replicate',
            padding=kernel_radius
        )
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
    ) -> Image.Image:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            tokenizer = self.pipeline.tokenizer
            text_encoder = self.pipeline.text_encoder
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

                image = Image.open(base_image_path).convert("RGB")
                image = t(image).to(
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )

                mask = Image.open(mask_image_path).convert("L")
                mask = t(mask).to(
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device,
                )

                erode_kernel = self.__create_erode_kernel(self.train_device)
                eroded_mask = erode_kernel(mask)
                eroded_mask = (eroded_mask > 0.5).float()

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
                latent_mask = (latent_mask > 0).float()
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
            tokenizer_output = tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.train_device)
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)
            else:
                tokens_attention_mask = None

            negative_tokenizer_output = tokenizer(
                negative_prompt,
                padding='max_length',
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            negative_tokens = negative_tokenizer_output.input_ids.to(self.train_device)
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                negative_tokens_attention_mask = negative_tokenizer_output.attention_mask.to(self.train_device)
            else:
                negative_tokens_attention_mask = None

            text_encoder_output = text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            final_layer_norm = text_encoder.text_model.final_layer_norm
            prompt_embedding = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
            )

            text_encoder_output = text_encoder(
                negative_tokens,
                attention_mask=negative_tokens_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            final_layer_norm = text_encoder.text_model.final_layer_norm
            negative_prompt_embedding = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
            )

            combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare timesteps
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            if force_last_timestep:
                last_timestep = torch.ones(1, device=self.train_device, dtype=torch.int64) \
                                * (noise_scheduler.config.num_train_timesteps - 1)

                # add the final timestep to force predicting with zero snr
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

            return image[0]

    def sample(
            self,
            sample_params: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: Callable[[Image.Image], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        prompt = self.model.add_embeddings_to_prompt(sample_params.prompt)
        negative_prompt = self.model.add_embeddings_to_prompt(sample_params.negative_prompt)

        if self.model_type.has_conditioning_image_input():
            image = self.__sample_inpainting(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=sample_params.height,
                width=sample_params.width,
                seed=sample_params.seed,
                random_seed=sample_params.random_seed,
                diffusion_steps=sample_params.diffusion_steps,
                cfg_scale=sample_params.cfg_scale,
                noise_scheduler=sample_params.noise_scheduler,
                cfg_rescale=0.7 if force_last_timestep else 0.0,
                sample_inpainting=sample_params.sample_inpainting,
                base_image_path=sample_params.base_image_path,
                mask_image_path=sample_params.mask_image_path,
                text_encoder_layer_skip=text_encoder_layer_skip,
                force_last_timestep=force_last_timestep,
                on_update_progress=on_update_progress,
            )
        else:
            image = self.__sample_base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=sample_params.height,
                width=sample_params.width,
                seed=sample_params.seed,
                random_seed=sample_params.random_seed,
                diffusion_steps=sample_params.diffusion_steps,
                cfg_scale=sample_params.cfg_scale,
                noise_scheduler=sample_params.noise_scheduler,
                cfg_rescale=0.7 if force_last_timestep else 0.0,
                text_encoder_layer_skip=text_encoder_layer_skip,
                force_last_timestep=force_last_timestep,
                on_update_progress=on_update_progress,
            )

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        image.save(destination)

        on_sample(image)
