from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
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
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseStableDiffusionXLSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attentions"],
        "attn-only": ["attn"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.unet)
            if model.unet_lora is not None:
                apply_circular_padding_to_conv2d(model.unet_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().unet,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().text_encoder_2,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().vae,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype, config)
        quantize_layers(model.text_encoder_2, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.vae_train_dtype, config)
        quantize_layers(model.unet, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state_1 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_1,
                    model.text_encoder_1,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][0][1:],
                )

                embedding_state_2 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_2,
                    model.text_encoder_2,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[1][0][1:],
                )
            else:
                embedding_state_1 = embedding_state.get("clip_l_out", embedding_state.get("clip_l", None))
                embedding_state_2 = embedding_state.get("clip_g_out", embedding_state.get("clip_g", None))

            embedding_state_1 = embedding_state_1.to(
                dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding_state_2 = embedding_state_2.to(
                dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionXLModelEmbedding(
                embedding_config.uuid,
                embedding_state_1,
                embedding_state_2,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer_1, model.all_text_encoder_1_embeddings())
        self._add_embeddings_to_tokenizer(model.tokenizer_2, model.all_text_encoder_2_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_1,
            orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_1_embeddings(),
        )
        model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_2,
            orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_2_embeddings(),
        )

        model.embedding_wrapper_1.hook_to_module()
        model.embedding_wrapper_2.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_text_encoder_1_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_1 = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_1)

        for embedding, embedding_config in zip(model.all_text_encoder_2_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_2 = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_2)

    def predict(
            self,
            model: StableDiffusionXLModel,
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

            text_encoder_output, pooled_text_encoder_2_output = model.combine_text_encoder_output(*model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens_1=batch['tokens_1'],
                tokens_2=batch['tokens_2'],
                text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                text_encoder_1_output=batch[
                    'text_encoder_1_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                text_encoder_2_output=batch[
                    'text_encoder_2_hidden_state'] if not config.train_text_encoder_2_or_embedding() else None,
                pooled_text_encoder_2_output=batch[
                    'text_encoder_2_pooled_state'] if not config.train_text_encoder_2_or_embedding() else None,
                text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
                text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
            ))

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

            # original size of the image
            original_height = batch['original_resolution'][0]
            original_width = batch['original_resolution'][1]
            crops_coords_top = batch['crop_offset'][0]
            crops_coords_left = batch['crop_offset'][1]
            target_height = batch['crop_resolution'][0]
            target_width = batch['crop_resolution'][1]

            add_time_ids = torch.stack([
                original_height,
                original_width,
                crops_coords_top,
                crops_coords_left,
                target_height,
                target_width
            ], dim=1)

            add_time_ids = add_time_ids.to(
                dtype=scaled_noisy_latent_image.dtype,
                device=scaled_noisy_latent_image.device,
            )

            if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
            predicted_latent_noise = model.unet(
                sample=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                added_cond_kwargs=added_cond_kwargs,
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

            if config.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens_1'], model.tokenizer_1),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    self._save_image(
                        self._project_latent_to_image_sdxl(latent_noise),
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step,
                        True
                    )

                    # predicted noise
                    self._save_image(
                        self._project_latent_to_image_sdxl(predicted_latent_noise),
                        config.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step,
                        True
                    )

                    # noisy image
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_noisy_latent_image),
                        config.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step,
                        True
                    )

                    # predicted image
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device)
                    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                        / sqrt_alpha_prod
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_predicted_latent_image),
                        config.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step,
                        True
                    )

                    # image
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_latent_image),
                        config.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step,
                        True
                    )

        model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
        return model_output_data

    def calculate_loss(
            self,
            model: StableDiffusionXLModel,
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

    def prepare_text_caching(self, model: StableDiffusionXLModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        if not config.train_text_encoder_2_or_embedding():
            model.text_encoder_2_to(self.train_device)

        model.eval()
        torch_gc()
