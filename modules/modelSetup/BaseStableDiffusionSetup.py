from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
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
    enable_checkpointing_for_clip_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseStableDiffusionSetup(
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
        "attn-mlp": ["attentions"],
        "attn-only": ["attn"],
        "full": [],
    }

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.vae.enable_gradient_checkpointing()
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.unet)
            if model.unet_lora is not None:
                apply_circular_padding_to_conv2d(model.unet_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().unet,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.unet, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer,
                    model.text_encoder,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][1:],
                )
            else:
                embedding_state = embedding_state.get("emp_params_out", embedding_state.get("emp_params", None))

            embedding_state = embedding_state.to(
                dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer,
            orig_module=model.text_encoder.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_embeddings(),
        )
        model.embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: StableDiffusionModel,
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
            model: StableDiffusionModel,
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

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch['tokens'],
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch[
                    'text_encoder_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
            )

            latent_noise = self._create_noise(
                scaled_latent_image,
                config,
                generator,
                timestep,
                model.noise_scheduler.betas,
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

            if config.model_type.has_depth_input():
                predicted_latent_noise = model.unet(
                    latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep,
                    text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    batch['latent_depth'].to(dtype=model.train_dtype.torch_dtype()),
                ).sample
            else:
                predicted_latent_noise = model.unet(
                    latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep,
                    text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                ).sample

            model_output_data = {}

            if model.noise_scheduler.config.prediction_type == 'epsilon':
                model_output_data = {
                    'loss_type': 'target',
                    'timestep': timestep,
                    'predicted': predicted_latent_noise,
                    'target': latent_noise,
                }
            elif model.noise_scheduler.config.prediction_type == 'v_prediction':
                target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
                model_output_data = {
                    'loss_type': 'target',
                    'timestep': timestep,
                    'predicted': predicted_latent_noise,
                    'target': target_velocity,
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
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device, dtype=model.vae.dtype)
                    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (
                                scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
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
            model: StableDiffusionModel,
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

    def prepare_text_caching(self, model: StableDiffusionModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
