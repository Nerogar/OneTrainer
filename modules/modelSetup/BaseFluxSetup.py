from abc import ABCMeta
from random import Random

from modules.model.FluxModel import FluxModel, FluxModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_clip_encoder_layers,
    enable_checkpointing_for_flux_transformer,
    enable_checkpointing_for_t5_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseFluxSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_flux_transformer(model.transformer, config)
            if model.text_encoder_1 is not None:
                enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
            if model.text_encoder_2 is not None:
                model.text_encoder_2_offload_conductor = \
                    enable_checkpointing_for_t5_encoder_layers(model.text_encoder_2, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().prior,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().text_encoder_2,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.text_encoder_2_autocast_context, model.text_encoder_2_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().text_encoder_2,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                    config.weight_dtypes().embedding if config.train_any_embedding() else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype)
        quantize_layers(model.text_encoder_2, self.train_device, model.text_encoder_2_train_dtype)
        quantize_layers(model.vae, self.train_device, model.train_dtype)
        quantize_layers(model.transformer, self.train_device, model.train_dtype)

    def _setup_embeddings(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                with model.autocast_context:
                    embedding_state_1 = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer_1,
                        model.text_encoder_1,
                    )

                    embedding_state_2 = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer_2,
                        model.text_encoder_2,
                        lambda text: model.encode_text(
                            text=text,
                            train_device=self.temp_device,
                        )[0][0][1:],
                    )
            else:
                embedding_state_1 = embedding_state.get("clip_l_out", embedding_state.get("clip_l", None))
                embedding_state_2 = embedding_state.get("t5_out", embedding_state.get("t5", None))

            if embedding_state_1 is not None:
                embedding_state_1 = embedding_state_1.to(
                    dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
                    device=self.train_device,
                ).detach()

            if embedding_state_2 is not None:
                embedding_state_2 = embedding_state_2.to(
                    dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
                    device=self.train_device,
                ).detach()

            embedding = FluxModelEmbedding(
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

        if model.tokenizer_1 is not None:
            self._add_embeddings_to_tokenizer(model.tokenizer_1, model.all_text_encoder_1_embeddings())
        if model.tokenizer_2 is not None:
            self._add_embeddings_to_tokenizer(model.tokenizer_2, model.all_text_encoder_2_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        if model.tokenizer_1 is not None and model.text_encoder_1 is not None:
            model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer_1,
                orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
                embeddings=model.all_text_encoder_1_embeddings(),
            )
        if model.tokenizer_2 is not None and model.text_encoder_2 is not None:
            model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer_2,
                orig_module=model.text_encoder_2.encoder.embed_tokens,
                embeddings=model.all_text_encoder_2_embeddings(),
            )

        if model.embedding_wrapper_1 is not None:
            model.embedding_wrapper_1.hook_to_module()
        if model.embedding_wrapper_2 is not None:
            model.embedding_wrapper_2.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        if model.text_encoder_1 is not None:
            for embedding, embedding_config in zip(model.all_text_encoder_1_embeddings(),
                                                   config.all_embedding_configs(), strict=True):
                train_embedding_1 = \
                    embedding_config.train \
                    and config.text_encoder.train_embedding \
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                embedding.requires_grad_(train_embedding_1)

        if model.text_encoder_2 is not None:
            for embedding, embedding_config in zip(model.all_text_encoder_2_embeddings(),
                                                   config.all_embedding_configs(), strict=True):
                train_embedding_2 = \
                    embedding_config.train \
                    and config.text_encoder_2.train_embedding \
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                embedding.requires_grad_(train_embedding_2)

    def predict(
            self,
            model: FluxModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            vae_scaling_factor = model.vae.config['scaling_factor']
            vae_shift_factor = model.vae.config['shift_factor']

            text_encoder_output, pooled_text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens_1=batch.get("tokens_1"),
                tokens_2=batch.get("tokens_2"),
                tokens_mask_2=batch.get("tokens_mask_2"),
                text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                pooled_text_encoder_1_output=batch['text_encoder_1_pooled_state'] \
                    if 'text_encoder_1_pooled_state' in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_2_output=batch['text_encoder_2_hidden_state'] \
                    if 'text_encoder_2_hidden_state' in batch and not config.train_text_encoder_2_or_embedding() else None,
                text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
                text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
                apply_attention_mask=config.prior.attention_mask,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = \
                    (batch['latent_conditioning_image'] - vae_shift_factor) * vae_scaling_factor

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                latent_height=scaled_latent_image.shape[-2],
                latent_width=scaled_latent_image.shape[-1],
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, scaled_latent_conditioning_image, batch['latent_mask']], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.prior.guidance_scale], device=self.train_device)
                guidance = guidance.expand(latent_input.shape[0])
            else:
                guidance = None

            text_ids = torch.zeros(
                size=(text_encoder_output.shape[1], 3),
                device=self.train_device,
            )

            image_ids = model.prepare_latent_image_ids(
                latent_input.shape[2],
                latent_input.shape[3],
                self.train_device,
                model.train_dtype.torch_dtype()
            )

            packed_latent_input = model.pack_latents(
                latent_input,
                latent_input.shape[0],
                latent_input.shape[1],
                latent_input.shape[2],
                latent_input.shape[3],
            )

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                guidance=guidance.to(dtype=model.train_dtype.torch_dtype()),
                pooled_projections=pooled_text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                txt_ids=text_ids.to(dtype=model.train_dtype.torch_dtype()),
                img_ids=image_ids.to(dtype=model.train_dtype.torch_dtype()),
                joint_attention_kwargs=None,
                return_dict=True
            ).sample

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                latent_input.shape[2],
                latent_input.shape[3],
            )

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
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
                        self._project_latent_to_image(latent_noise),
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step,
                    )

                    # noisy image
                    self._save_image(
                        self._project_latent_to_image(scaled_noisy_latent_image),
                        config.debug_dir + "/training_batches",
                        "2-noisy_image",
                        train_progress.global_step,
                    )

                    # predicted flow
                    self._save_image(
                        self._project_latent_to_image(predicted_flow),
                        config.debug_dir + "/training_batches",
                        "3-predicted_flow",
                        train_progress.global_step,
                    )

                    # flow
                    flow = latent_noise - scaled_latent_image
                    self._save_image(
                        self._project_latent_to_image(flow),
                        config.debug_dir + "/training_batches",
                        "4-flow",
                        train_progress.global_step,
                    )

                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma

                    # predicted image
                    self._save_image(
                        self._project_latent_to_image(predicted_scaled_latent_image),
                        config.debug_dir + "/training_batches",
                        "5-predicted_image",
                        train_progress.global_step,
                    )

                    # image
                    self._save_image(
                        self._project_latent_to_image(scaled_latent_image),
                        config.debug_dir + "/training_batches",
                        "6-image",
                        model.train_progress.global_step,
                    )

        return model_output_data

    def calculate_loss(
            self,
            model: FluxModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas.to(device=self.train_device),
        ).mean()
