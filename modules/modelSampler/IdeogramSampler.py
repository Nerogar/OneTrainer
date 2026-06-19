import copy
from collections.abc import Callable

from modules.model.IdeogramModel import IdeogramModel
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

from diffusers.pipelines.ideogram4.pipeline_ideogram4 import _logit_normal_sigmas, _resolution_aware_mu

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


@factory.register(BaseModelSampler, ModelType.IDEOGRAM_4)
class IdeogramSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: IdeogramModel,
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
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            vae = self.pipeline.vae
            transformer = self.pipeline.transformer
            dtype = self.model.train_dtype.torch_dtype()

            # Ideogram uses asymmetric (dual-network) CFG: the negative branch is the unconditional_transformer run on
            # the image tokens with zeroed text features, NOT a negative-prompt encode. negative_prompt is unused.
            use_cfg = cfg_scale > 1.0
            if use_cfg and self.model.unconditional_transformer is None:
                raise RuntimeError(
                    "cfg_scale > 1.0 requires the unconditional transformer, which was not loaded. "
                    "Either set cfg_scale to 1.0 or enable loading the unconditional transformer."
                )

            vae_scale_factor = 8
            patch_size = 2
            latent_dim = transformer.config.in_channels
            grid_h = height // (vae_scale_factor * patch_size)
            grid_w = width // (vae_scale_factor * patch_size)
            num_image_tokens = grid_h * grid_w

            # encode text (conditional branch only)
            self.model.text_encoder_to(self.train_device)
            text_features, text_lengths = self.model.encode_text(
                train_device=self.train_device,
                text=prompt,
            )
            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # build the packed [text][image] conditioning (no left pad for a single prompt). Padding positions are
            # masked out by segment_ids/indicator, so packing to the actual text length matches the 2048-pad pipeline.
            max_text_tokens = text_features.shape[1]
            position_ids, segment_ids, indicator = self.model.prepare_packed_ids(
                text_lengths, grid_h, grid_w, max_text_tokens, self.train_device,
            )
            image_feature_padding = torch.zeros(
                text_features.shape[0], num_image_tokens, text_features.shape[-1],
                dtype=text_features.dtype, device=self.train_device,
            )
            llm_features = torch.cat([text_features, image_feature_padding], dim=1).to(dtype)
            text_z_padding = torch.zeros(
                text_features.shape[0], max_text_tokens, latent_dim, dtype=dtype, device=self.train_device,
            )

            # unconditional (image-only) branch: zeroed text features over the image-region slices of the layout
            if use_cfg:
                neg_position_ids = position_ids[:, max_text_tokens:]
                neg_segment_ids = segment_ids[:, max_text_tokens:]
                neg_indicator = indicator[:, max_text_tokens:]
                neg_llm_features = torch.zeros(
                    text_features.shape[0], num_image_tokens, text_features.shape[-1],
                    dtype=dtype, device=self.train_device,
                )

            # packed (B, num_image_tokens, latent_dim) noise
            latent_image = torch.randn(
                size=(text_features.shape[0], num_image_tokens, latent_dim),
                generator=generator, device=self.train_device, dtype=torch.float32,
            )

            # both are dead after this point but stay referenced as locals for the rest of the function (the
            # denoising loop + VAE decode); image_feature_padding alone is ~832MB fp32 at 1024px, freeing it here
            # closes the gap on the OOM observed in llm_cond_norm's fp32 variance upcast of neg_llm_features.
            del image_feature_padding, text_features

            # resolution-aware logit-normal Euler schedule (pipeline overrides the scheduler's default sigmas)
            schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=0.0)
            sigmas = _logit_normal_sigmas(diffusion_steps, schedule_mu, std=1.5, device=self.train_device)
            noise_scheduler.set_timesteps(sigmas=sigmas.tolist(), device=self.train_device)
            timesteps = noise_scheduler.timesteps
            num_train_timesteps = noise_scheduler.config.num_train_timesteps

            self.model.transformer_to(self.train_device)
            if use_cfg:
                self.model.unconditional_transformer_to(self.train_device)

            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                # scheduler stores num_train_timesteps-scaled timesteps; convert back to model time (0=noise, 1=data)
                t_model = (1.0 - timestep.float() / num_train_timesteps).expand(latent_image.shape[0])

                pos_z = torch.cat([text_z_padding, latent_image.to(dtype)], dim=1)
                pos_out = transformer(
                    hidden_states=pos_z,
                    timestep=t_model,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    return_dict=False,
                )[0]
                pos_v = pos_out[:, max_text_tokens:].float()

                if use_cfg:
                    neg_v = self.model.unconditional_transformer(
                        hidden_states=latent_image.to(dtype),
                        timestep=t_model,
                        encoder_hidden_states=neg_llm_features,
                        position_ids=neg_position_ids,
                        segment_ids=neg_segment_ids,
                        indicator=neg_indicator,
                        return_dict=False,
                    )[0].float()
                    v = neg_v + cfg_scale * (pos_v - neg_v)
                else:
                    v = pos_v

                latent_image = noise_scheduler.step(-v, timestep, latent_image, return_dict=False)[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            self.model.unconditional_transformer_to(self.temp_device)
            torch_gc()
            self.model.vae_to(self.train_device)

            # bn-denormalize the packed latents and unpatchify back to (B, C, H, W) before VAE decode
            latents = self.model.unscale_latents(latent_image)
            latents = self.model.unpatchify_latents(latents, grid_h, grid_w)

            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            # no VaeImageProcessor — match the pipeline's manual postprocess
            image = (image.clamp(-1, 1) + 1) / 2
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = [PILImage.fromarray((img * 255).astype(np.uint8)) for img in image]

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
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
