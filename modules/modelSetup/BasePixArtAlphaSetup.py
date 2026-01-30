from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.PixArtAlphaModel import PixArtAlphaModel, PixArtAlphaModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionMixin import ModelSetupDiffusionMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_basic_transformer_blocks,
    enable_checkpointing_for_t5_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BasePixArtAlphaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn1", "attn2", "ff.net"],
        "attn-only": ["attn1", "attn2"],
        "blocks": ["transformer_block"],
        "full": [],
    }

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.vae.enable_gradient_checkpointing()
            model.transformer_offload_conductor = \
                enable_checkpointing_for_basic_transformer_blocks(model.transformer, config, offload_enabled=True)
            model.text_encoder_offload_conductor = \
                enable_checkpointing_for_t5_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                config.weight_dtypes().embedding if config.train_any_embedding() else None,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                with model.autocast_context:
                    embedding_state = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer,
                        model.text_encoder,
                        lambda text: model.encode_text(
                            text=text,
                            train_device=self.temp_device,
                        )[0][0][1:],
                    )
            else:
                embedding_state = embedding_state.get("t5_out", embedding_state.get("t5", None))

            embedding_state = embedding_state.to(
                dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = PixArtAlphaModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding),

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer,
            orig_module=model.text_encoder.encoder.embed_tokens,
            embeddings=model.all_text_encoder_embeddings(),
        )
        model.embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_text_encoder_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = \
                embedding_config.train \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding)

    def predict(
            self,
            model: PixArtAlphaModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            vae_scaling_factor = model.vae.config['scaling_factor']

            text_encoder_output, text_encoder_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch['tokens'],
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch[
                    'text_encoder_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                attention_mask=batch['tokens_mask'],
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
            )

            scaled_noisy_latent_image = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.betas,
            )

            if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            batch_size = latent_input.shape[0]
            height = latent_input.shape[2] * 8
            width = latent_input.shape[3] * 8
            resolution = torch.tensor([height, width]).repeat(batch_size, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
            resolution = resolution.to(dtype=config.train_dtype.torch_dtype(), device=self.train_device)
            aspect_ratio = aspect_ratio.to(dtype=config.train_dtype.torch_dtype(), device=self.train_device)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            text_encoder_attention_mask = text_encoder_attention_mask.view(batch_size, -1)

            predicted_latent_noise, predicted_latent_var_values = model.transformer(
                latent_input.to(dtype=config.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=config.train_dtype.torch_dtype()),
                encoder_attention_mask=text_encoder_attention_mask.to(dtype=config.train_dtype.torch_dtype()),
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
            ).sample.chunk(2, dim=1)

            model_output_data = {
                'loss_type': 'target',
                'predicted': predicted_latent_noise,
                'target': latent_noise,
                'noisy_latent_image': scaled_noisy_latent_image,
                'predicted_var_values': predicted_latent_var_values,
                'timestep': timestep,
                'scaled_latent_image': scaled_latent_image,
            }

            if self.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.tokenizer),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    noise = model.vae.decode(latent_noise / vae_scaling_factor).sample
                    noise = noise.clamp(-1, 1)
                    self._save_image(
                        noise,
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # predicted noise
                    predicted_noise = model.vae.decode(predicted_latent_noise / vae_scaling_factor).sample
                    predicted_noise = predicted_noise.clamp(-1, 1)
                    self._save_image(
                        predicted_noise,
                        config.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step
                    )

                    # noisy image
                    noisy_latent_image = scaled_noisy_latent_image / vae_scaling_factor
                    noisy_image = model.vae.decode(noisy_latent_image).sample
                    noisy_image = noisy_image.clamp(-1, 1)
                    self._save_image(
                        noisy_image,
                        config.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step
                    )

                    # predicted image
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device)
                    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (scaled_noisy_latent_image - predicted_latent_noise
                         * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                    predicted_latent_image = scaled_predicted_latent_image / vae_scaling_factor
                    predicted_image = model.vae.decode(predicted_latent_image).sample
                    predicted_image = predicted_image.clamp(-1, 1)
                    self._save_image(
                        predicted_image,
                        config.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step
                    )

                    # image
                    image = model.vae.decode(latent_image).sample
                    image = image.clamp(-1, 1)
                    self._save_image(
                        image,
                        config.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step
                    )

                    # conditioning image
                    if config.model_type.has_conditioning_image_input():
                        conditioning_image = model.vae.decode(
                            scaled_latent_conditioning_image / vae_scaling_factor).sample
                        conditioning_image = conditioning_image.clamp(-1, 1)
                        self._save_image(
                            conditioning_image,
                            config.debug_dir + "/training_batches",
                            "6-conditioning_image",
                            train_progress.global_step
                        )

        model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
        return model_output_data

    def calculate_loss(
            self,
            model: PixArtAlphaModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            betas=model.noise_scheduler.betas,
        ).mean()

    def prepare_text_caching(self, model: PixArtAlphaModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
