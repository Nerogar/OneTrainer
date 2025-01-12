from abc import ABCMeta
from random import Random

from modules.model.HunyuanVideoModel import HunyuanVideoModel, HunyuanVideoModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_clip_encoder_layers,
    enable_checkpointing_for_hunyuan_video_transformer,
    enable_checkpointing_for_llama_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseHunyuanVideoSetup(
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
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        # if config.attention_mechanism == AttentionMechanism.DEFAULT:
        #     model.transformer.set_attn_processor(HunyuanVideoAttnProcessor2_0())
        # elif config.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
        #     try:
        #         model.transformer.set_attn_processor(HunyuanVideoXFormersAttnProcessor(model.train_dtype.torch_dtype()))
        #         model.vae.enable_xformers_memory_efficient_attention()
        #     except Exception as e:
        #         print(
        #             "Could not enable memory efficient attention. Make sure xformers is installed"
        #             f" correctly and a GPU is available: {e}"
        #         )
        # elif config.attention_mechanism == AttentionMechanism.SDP:
        #     model.transformer.set_attn_processor(HunyuanVideoAttnProcessor2_0())
        #
        #     if is_xformers_available():
        #         try:
        #             model.vae.enable_xformers_memory_efficient_attention()
        #         except Exception as e:
        #             print(
        #                 "Could not enable memory efficient attention. Make sure xformers is installed"
        #                 f" correctly and a GPU is available: {e}"
        #             )

        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_hunyuan_video_transformer(model.transformer, config)
            if model.text_encoder_1 is not None:
                model.text_encoder_1_offload_conductor = \
                    enable_checkpointing_for_llama_encoder_layers(model.text_encoder_1, config)
            if model.text_encoder_2 is not None:
                enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

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

        model.transformer_autocast_context, model.transformer_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().prior,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                    config.weight_dtypes().embedding if config.train_any_embedding() else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype)
        quantize_layers(model.text_encoder_2, self.train_device, model.train_dtype)
        quantize_layers(model.vae, self.train_device, model.train_dtype)
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype)

        model.vae.enable_tiling()

    def _setup_additional_embeddings(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        model.additional_embeddings = []
        for i, embedding_config in enumerate(config.additional_embeddings):
            embedding_state = model.additional_embedding_states[i]
            if embedding_state is None:
                if model.tokenizer_1 is not None and model.text_encoder_1 is not None:
                    embedding_state_1 = self._create_new_embedding(
                        model.tokenizer_1,
                        model.text_encoder_1,
                        config.additional_embeddings[i].initial_embedding_text,
                        config.additional_embeddings[i].token_count,
                    )
                else:
                    embedding_state_1 = None

                if model.tokenizer_2 is not None and model.text_encoder_2 is not None:
                    embedding_state_2 = self._create_new_embedding(
                        model.tokenizer_2,
                        model.text_encoder_2,
                        config.additional_embeddings[i].initial_embedding_text,
                        config.additional_embeddings[i].token_count,
                    )
                else:
                    embedding_state_2 = None
            else:
                embedding_state_1, embedding_state_2 = embedding_state

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

            embedding = HunyuanVideoModelEmbedding(
                embedding_config.uuid,
                embedding_state_1,
                embedding_state_2,
                embedding_config.placeholder,
            )
            model.additional_embeddings.append(embedding)
            if model.tokenizer_1 is not None:
                self._add_embedding_to_tokenizer(model.tokenizer_1, embedding.text_tokens)
            if model.tokenizer_2 is not None:
                self._add_embedding_to_tokenizer(model.tokenizer_2, embedding.text_tokens)

    def _setup_embedding(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        model.embedding = None

        embedding_state = model.embedding_state
        if embedding_state is None:
            if model.tokenizer_1 is not None and model.text_encoder_1 is not None:
                embedding_state_1 = self._create_new_embedding(
                    model.tokenizer_1,
                    model.text_encoder_1,
                    config.embedding.initial_embedding_text,
                    config.embedding.token_count,
                )
            else:
                embedding_state_1 = None

            if model.tokenizer_2 is not None and model.text_encoder_2 is not None:
                embedding_state_2 = self._create_new_embedding(
                    model.tokenizer_2,
                    model.text_encoder_2,
                    config.embedding.initial_embedding_text,
                    config.embedding.token_count,
                )
            else:
                embedding_state_2 = None
        else:
            embedding_state_1, embedding_state_2 = embedding_state

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

        model.embedding = HunyuanVideoModelEmbedding(
            config.embedding.uuid,
            embedding_state_1,
            embedding_state_2,
            config.embedding.placeholder,
        )
        if model.tokenizer_1 is not None:
            self._add_embedding_to_tokenizer(model.tokenizer_1, model.embedding.text_tokens)
        if model.tokenizer_2 is not None:
            self._add_embedding_to_tokenizer(model.tokenizer_2, model.embedding.text_tokens)

    def _setup_embedding_wrapper(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        if model.tokenizer_1 is not None and model.text_encoder_1 is not None:
            model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer_1,
                orig_module=model.text_encoder_1.embed_tokens,
                additional_embeddings=[embedding.text_encoder_1_vector for embedding in model.additional_embeddings]
                                      + ([] if model.embedding is None else [model.embedding.text_encoder_1_vector]),
                additional_embedding_placeholders=[embedding.placeholder for embedding in model.additional_embeddings]
                                                  + ([] if model.embedding is None else [model.embedding.placeholder]),
                additional_embedding_names=[embedding.uuid for embedding in model.additional_embeddings]
                                           + ([] if model.embedding is None else [model.embedding.uuid]),
            )
        if model.tokenizer_2 is not None and model.text_encoder_2 is not None:
            model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer_2,
                orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
                additional_embeddings=[embedding.text_encoder_2_vector for embedding in model.additional_embeddings]
                                      + ([] if model.embedding is None else [model.embedding.text_encoder_2_vector]),
                additional_embedding_placeholders=[embedding.placeholder for embedding in model.additional_embeddings]
                                                  + ([] if model.embedding is None else [model.embedding.placeholder]),
                additional_embedding_names=[embedding.uuid for embedding in model.additional_embeddings]
                                           + ([] if model.embedding is None else [model.embedding.uuid]),
            )

        if model.embedding_wrapper_1 is not None:
            model.embedding_wrapper_1.hook_to_module()
        if model.embedding_wrapper_2 is not None:
            model.embedding_wrapper_2.hook_to_module()

    def predict(
            self,
            model: HunyuanVideoModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(train_progress.global_step)
            rand = Random(train_progress.global_step)

            vae_scaling_factor = model.vae.config['scaling_factor']

            text_encoder_output, pooled_text_encoder_output, text_encoder_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens_1=batch.get("tokens_1"),
                tokens_2=batch.get("tokens_2"),
                tokens_mask_1=batch.get("tokens_mask_1"),
                text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                text_encoder_1_output=batch['text_encoder_1_hidden_state'] \
                    if 'text_encoder_1_hidden_state' in batch and not config.train_text_encoder_or_embedding() else None,
                pooled_text_encoder_2_output=batch['text_encoder_2_pooled_state'] \
                    if 'text_encoder_2_pooled_state' in batch and not config.train_text_encoder_2_or_embedding() else None,
                text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
                text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            if scaled_latent_image.ndim == 4:
                scaled_latent_image = scaled_latent_image.unsqueeze(2)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            latent_input = scaled_noisy_latent_image

            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.prior.guidance_scale * 1000.0], device=self.train_device)
                guidance = guidance.expand(latent_input.shape[0])
            else:
                guidance = None

            with model.transformer_autocast_context:
                predicted_flow = model.transformer(
                    hidden_states=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=timestep,
                    guidance=guidance.to(dtype=model.train_dtype.torch_dtype()),
                    pooled_projections=pooled_text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    encoder_attention_mask=text_encoder_attention_mask.to(dtype=model.train_dtype.torch_dtype()),
                    return_dict=True
                ).sample

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
            model: HunyuanVideoModel,
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
