import inspect
from collections.abc import Callable

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import create, factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.image_util import load_image
from modules.util.staged_pipeline import run_staged_pipeline
from modules.util.tqdm_util import tqdm

import torch
from torch import nn
from torchvision.transforms import transforms


@factory.register(BaseModelSampler, ModelType.STABLE_DIFFUSION_XL_10_BASE)
@factory.register(BaseModelSampler, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
class StableDiffusionXLSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: StableDiffusionXLModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

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

    # only present for conditioning (inpainting) model types: VAE-encode the conditioning image + mask
    @torch.no_grad()
    def __cond_encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        self.model.materialize_only("vae")
        vae = self.pipeline.vae
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)

        with self.model.vae_autocast_context:
            if sample_config.sample_inpainting:
                t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(
                        (height, width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                    ),
                ])

                image = load_image(sample_config.base_image_path, convert_mode="RGB")
                image = t(image).to(
                    dtype=self.model.vae_train_dtype.torch_dtype(),
                    device=self.train_device,
                )

                mask = load_image(sample_config.mask_image_path, convert_mode='L')
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
                    dtype=self.model.vae_train_dtype.torch_dtype(),
                    device=self.train_device,
                )
                latent_conditioning_image = vae.encode(conditioning_image).latent_dist.mode() * vae.config.scaling_factor
                latent_mask = torch.ones(
                    size=(1, 1, latent_conditioning_image.shape[2], latent_conditioning_image.shape[3]),
                    dtype=self.model.train_dtype.torch_dtype(),
                    device=self.train_device
                )

        return {
            "latent_conditioning_image": latent_conditioning_image,
            "latent_mask": latent_mask,
        }

    @torch.no_grad()
    def __encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        self.model.materialize_only_text_encoders()
        prompt_embedding, pooled_text_encoder_2_output = self.model.combine_text_encoder_output(
            *self.model.encode_text(
                text=sample_config.prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
            ))

        negative_prompt_embedding, negative_pooled_text_encoder_2_output = self.model.combine_text_encoder_output(
            *self.model.encode_text(
                text=sample_config.negative_prompt,
                train_device=self.train_device,
                text_encoder_1_layer_skip=sample_config.text_encoder_1_layer_skip,
                text_encoder_2_layer_skip=sample_config.text_encoder_2_layer_skip,
            ))

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding]) \
            .to(dtype=self.model.train_dtype.torch_dtype())

        return {
            "combined_prompt_embedding": combined_prompt_embedding,
            "pooled_text_encoder_2_output": pooled_text_encoder_2_output,
            "negative_pooled_text_encoder_2_output": negative_pooled_text_encoder_2_output,
        }

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            combined_prompt_embedding: torch.Tensor,
            pooled_text_encoder_2_output: torch.Tensor,
            negative_pooled_text_encoder_2_output: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
            latent_conditioning_image: torch.Tensor | None = None,
            latent_mask: torch.Tensor | None = None,
    ) -> dict:
        self.model.materialize_only("unet")
        # conditioning tensors are only present for inpainting model types (their cond-encode stage ran)
        is_inpainting = latent_conditioning_image is not None
        unet = self.pipeline.unet
        vae_scale_factor = self.pipeline.vae_scale_factor
        height = self.quantize_resolution(sample_config.height, 64)
        width = self.quantize_resolution(sample_config.width, 64)
        cfg_scale = sample_config.cfg_scale
        cfg_rescale = 0.7 if sample_config.force_last_timestep else 0.0
        force_last_timestep = sample_config.force_last_timestep
        diffusion_steps = sample_config.diffusion_steps

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = create.create_noise_scheduler(sample_config.noise_scheduler, self.model.noise_scheduler, diffusion_steps)

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

        original_height = height
        original_width = width
        crops_coords_top = 0
        crops_coords_left = 0
        target_height = height
        target_width = width

        add_time_ids = torch.tensor([
            original_height,
            original_width,
            crops_coords_top,
            crops_coords_left,
            target_height,
            target_width
        ]).unsqueeze(dim=0)

        add_time_ids = add_time_ids.to(
            device=self.train_device,
        )

        # prepare latent image
        num_channels_latents = latent_conditioning_image.shape[1] if is_inpainting else unet.config.in_channels
        latent_image = torch.randn(
            size=(1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=self.train_device,
            dtype=self.model.train_dtype.torch_dtype(),
        )

        if is_inpainting and sample_config.sample_inpainting:
            # SDXL inpainting is terrible at reconstructing from pure noise.
            # This removes the last timestep to let the model know about the general image composition and brightness
            timesteps = timesteps[1:]
            latent_image = noise_scheduler.add_noise(latent_conditioning_image, latent_image, timesteps[:1])
        else:
            latent_image = latent_image * noise_scheduler.init_noise_sigma

        added_cond_kwargs = {
            "text_embeds": torch.concat([negative_pooled_text_encoder_2_output, pooled_text_encoder_2_output], dim=0),
            "time_ids": torch.concat([add_time_ids] * 2, dim=0),
        }

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        for i, timestep in enumerate(tqdm(timesteps, desc="steps", leave=False)):
            if is_inpainting:
                latent_model_input = noise_scheduler.scale_model_input(latent_image, timestep)
                latent_model_input = torch.concat(
                    [latent_model_input, latent_mask, latent_conditioning_image], 1
                )
                latent_model_input = torch.cat([latent_model_input] * 2)
            else:
                latent_model_input = torch.cat([latent_image] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            noise_pred = unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=combined_prompt_embedding,
                added_cond_kwargs=added_cond_kwargs,
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

        latent_image = latent_image.to(dtype=self.model.vae_train_dtype.torch_dtype())
        with self.model.vae_autocast_context:
            image = vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]

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
        stages = [("encoding", self.__encode), ("denoising", self.__denoise), ("decoding", self.__decode)]
        # conditioning (inpainting) model types VAE-encode a conditioning image first
        if self.model_type.has_conditioning_image_input():
            stages.insert(0, ("encoding conditioning image", self.__cond_encode))

        batch_progress = self.batch_progress_callback(sample_configs, on_update_progress)

        with self.model.autocast_context:
            sampler_outputs = run_staged_pipeline(
                stages,
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
