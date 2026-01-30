import inspect
from collections.abc import Callable

from modules.model.WuerstchenModel import WuerstchenModel
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


class WuerstchenSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: WuerstchenModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    def __sample_prior(
            self,
            prompt,
            negative_prompt,
            height,
            width,
            generator,
            diffusion_steps,
            cfg_scale,
            text_encoder_layer_skip,
            prior_noise_scheduler,
            prior_prior,
            on_update_progress,
    ):
        # prepare prompt
        self.model.prior_text_encoder_to(self.train_device)

        prompt_embedding, pooled_prompt_embedding = self.model.encode_text(
            text=prompt,
            train_device=self.train_device,
            text_encoder_layer_skip=text_encoder_layer_skip,
        )

        negative_prompt_embedding, pooled_negative_prompt_embedding = self.model.encode_text(
            text=negative_prompt,
            train_device=self.train_device,
            text_encoder_layer_skip=text_encoder_layer_skip,
        )

        combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding]) \
            .to(dtype=self.model.prior_train_dtype.torch_dtype())
        if self.model_type.is_stable_cascade():
            combined_pooled_prompt_embedding = torch.cat([pooled_negative_prompt_embedding, pooled_prompt_embedding]) \
                .to(dtype=self.model.prior_train_dtype.torch_dtype())

        self.model.prior_text_encoder_to(self.temp_device)
        torch_gc()

        # prepare timesteps
        prior_noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
        timesteps = prior_noise_scheduler.timesteps

        # prepare latent image
        num_channels_latents = 16
        latent_width = int((width * 0.75) / 32.0)
        latent_height = int((height * 0.75) / 32.0)
        latent_image = torch.randn(
            size=(1, num_channels_latents, latent_height, latent_width),
            generator=generator,
            device=self.train_device,
            dtype=self.model.prior_train_dtype.torch_dtype(),
        ) * prior_noise_scheduler.init_noise_sigma

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(prior_noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        clip_img = torch.zeros(size=(2, 1, 768), dtype=self.model.prior_train_dtype.torch_dtype(), device=combined_prompt_embedding.device)

        self.model.prior_prior_to(self.train_device)
        for i, timestep in enumerate(tqdm(timesteps[:-1], desc="sampling")):
            timestep = torch.stack([timestep]).to(dtype=self.model.prior_train_dtype.torch_dtype())

            latent_model_input = torch.cat([latent_image] * 2)

            # predict the noise residual
            with self.model.prior_autocast_context:
                if self.model_type.is_wuerstchen_v2():
                    prior_kwargs = {
                        'c': combined_prompt_embedding,
                    }
                elif self.model_type.is_stable_cascade():
                    prior_kwargs = {
                        'clip_text': combined_prompt_embedding,
                        'clip_text_pooled': combined_pooled_prompt_embedding,
                        'clip_img': clip_img,
                    }

                noise_pred = prior_prior(
                    latent_model_input,
                    torch.cat([timestep] * 2),
                    **prior_kwargs,
                )

                if self.model.model_type.is_stable_cascade():
                    noise_pred = noise_pred.sample

            # cfg
            noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)

            # compute the previous noisy sample x_t -> x_t-1
            latent_image = prior_noise_scheduler.step(
                noise_pred, timestep, latent_image, **extra_step_kwargs
            ).prev_sample

            on_update_progress(i + 1, len(timesteps))

        self.model.prior_prior_to(self.temp_device)
        torch_gc()

        if self.model_type.is_wuerstchen_v2():
            latent_image = latent_image * 42.0 - 1.0

        latent_image = latent_image.to(dtype=self.model.prior_train_dtype.torch_dtype())

        return latent_image

    def __sample_decoder(
            self,
            prompt,
            height,
            width,
            generator,
            diffusion_steps,
            text_encoder_layer_skip,
            image_embedding,
            decoder_tokenizer,
            decoder_text_encoder,
            decoder_noise_scheduler,
            decoder_decoder,
            on_update_progress,
    ):
        # prepare prompt
        if self.model_type.is_wuerstchen_v2():
            self.model.decoder_text_encoder_to(self.train_device)
        elif self.model_type.is_stable_cascade():
            self.model.prior_text_encoder_to(self.train_device)
        tokenizer_output = decoder_tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=decoder_tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = tokenizer_output.input_ids.to(self.train_device)
        tokens_attention_mask = tokenizer_output.attention_mask.to(self.train_device)

        text_encoder_output = decoder_text_encoder(
            tokens,
            attention_mask=tokens_attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        final_layer_norm = decoder_text_encoder.text_model.final_layer_norm
        prompt_embedding = final_layer_norm(
            text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
        )

        if self.model_type.is_stable_cascade():
            prompt_embedding = text_encoder_output.text_embeds.unsqueeze(1)

        if self.model_type.is_wuerstchen_v2():
            self.model.decoder_text_encoder_to(self.temp_device)
        elif self.model_type.is_stable_cascade():
            self.model.prior_text_encoder_to(self.temp_device)
        torch_gc()

        # prepare timesteps
        decoder_noise_scheduler.set_timesteps(10, device=self.train_device)
        timesteps = decoder_noise_scheduler.timesteps

        # prepare latent image
        num_channels_latents = 4
        latent_width = width // 4
        latent_height = height // 4
        latent_image = torch.randn(
            size=(1, num_channels_latents, latent_height, latent_width),
            generator=generator,
            device=self.train_device,
            dtype=self.model.prior_train_dtype.torch_dtype(),
        ) * decoder_noise_scheduler.init_noise_sigma

        # denoising loop
        extra_step_kwargs = {}
        if "generator" in set(inspect.signature(decoder_noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator

        self.model.decoder_decoder_to(self.train_device)
        for i, timestep in enumerate(tqdm(timesteps[:-1], desc="sampling")):
            timestep = torch.stack([timestep]).to(dtype=self.model.prior_train_dtype.torch_dtype())

            latent_model_input = latent_image

            # predict the noise residual
            if self.model_type.is_wuerstchen_v2():
                decoder_kwargs = {
                    'effnet': image_embedding,
                    'clip': prompt_embedding,
                }
            elif self.model_type.is_stable_cascade():
                decoder_kwargs = {
                    'clip_text_pooled': prompt_embedding,
                    'effnet': image_embedding,
                }

            noise_pred = decoder_decoder(
                latent_model_input,
                timestep,
                **decoder_kwargs,
            )

            if self.model.model_type.is_stable_cascade():
                noise_pred = noise_pred.sample

            # compute the previous noisy sample x_t -> x_t-1
            latent_image = decoder_noise_scheduler.step(
                noise_pred, timestep, latent_image, **extra_step_kwargs
            ).prev_sample

            on_update_progress(i + 1, len(timesteps))

        self.model.decoder_decoder_to(self.temp_device)
        torch_gc()

        return latent_image

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
        generator = torch.Generator(device=self.train_device)
        if random_seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        height = (height // 128) * 128
        width = (width // 128) * 128

        prior_noise_scheduler = self.model.prior_noise_scheduler
        prior_prior = self.model.prior_prior

        if self.model_type.is_wuerstchen_v2():
            decoder_tokenizer = self.model.decoder_tokenizer
            decoder_text_encoder = self.model.decoder_text_encoder
        elif self.model_type.is_stable_cascade():
            decoder_tokenizer = self.model.prior_tokenizer
            decoder_text_encoder = self.model.prior_text_encoder

        decoder_noise_scheduler = self.model.decoder_noise_scheduler
        decoder_decoder = self.model.decoder_decoder
        decoder_vqgan = self.model.decoder_vqgan

        with self.model.autocast_context:
            image_embedding = self.__sample_prior(
                prompt,
                negative_prompt,
                height,
                width,
                generator,
                diffusion_steps,
                cfg_scale,
                text_encoder_layer_skip,
                prior_noise_scheduler,
                prior_prior,
                on_update_progress,
            )

            latent_image = self.__sample_decoder(
                prompt,
                height,
                width,
                generator,
                diffusion_steps,
                text_encoder_layer_skip,
                image_embedding,
                decoder_tokenizer,
                decoder_text_encoder,
                decoder_noise_scheduler,
                decoder_decoder,
                on_update_progress,
            )

            # decode vqgan
            self.model.decoder_vqgan_to(self.train_device)

            latents = decoder_vqgan.config.scale_factor * latent_image
            image_tensor = decoder_vqgan.decode(latents).sample.clamp(0, 1)
            image_array = image_tensor.permute(0, 2, 3, 1).cpu().squeeze().float().numpy()
            image_array = (image_array * 255).round().astype("uint8")

            self.model.decoder_vqgan_to(self.temp_device)
            torch_gc()

        return ModelSamplerOutput(
            file_type=FileType.IMAGE,
            data=Image.fromarray(image_array),
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
            height=self.quantize_resolution(sample_config.height, 128),
            width=self.quantize_resolution(sample_config.width, 128),
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

factory.register(BaseModelSampler, WuerstchenSampler, ModelType.WUERSTCHEN_2)
factory.register(BaseModelSampler, WuerstchenSampler, ModelType.STABLE_CASCADE_1)
