from abc import ABCMeta
from random import Random

from modules.model.SanaModel import SanaModel, SanaModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_gemma_layers,
    enable_checkpointing_for_sana_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseSanaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta,
):

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        # if config.attention_mechanism == AttentionMechanism.DEFAULT:
        #     for child_module in model.transformer.modules():
        #         if isinstance(child_module, Attention):
        #             child_module.set_processor(AttnProcessor())
        # elif config.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
        #     try:
        #         for child_module in model.transformer.modules():
        #             if isinstance(child_module, Attention):
        #                 child_module.set_processor(XFormersAttnProcessor())
        #         model.vae.enable_xformers_memory_efficient_attention()
        #     except Exception as e:
        #         print(
        #             "Could not enable memory efficient attention. Make sure xformers is installed"
        #             f" correctly and a GPU is available: {e}"
        #         )
        # elif config.attention_mechanism == AttentionMechanism.SDP:
        #     for child_module in model.transformer.modules():
        #         if isinstance(child_module, Attention):
        #             child_module.set_processor(AttnProcessor2_0())
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
            # model.vae.enable_gradient_checkpointing()
            model.transformer_offload_conductor = \
                enable_checkpointing_for_sana_transformer(model.transformer, config)
            model.text_encoder_offload_conductor = \
                enable_checkpointing_for_gemma_layers(model.text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().prior,
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

        model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().vae,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype)
        quantize_layers(model.vae, self.train_device, model.train_dtype)
        quantize_layers(model.transformer, self.train_device, model.train_dtype)

    def _setup_additional_embeddings(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        model.additional_embeddings = []
        for i, embedding_config in enumerate(config.additional_embeddings):
            embedding_state = model.additional_embedding_states[i]
            if embedding_state is None:
                embedding_state = self._create_new_embedding(
                    model.tokenizer,
                    model.text_encoder,
                    config.additional_embeddings[i].initial_embedding_text,
                    config.additional_embeddings[i].token_count,
                )

            embedding_state = embedding_state.to(
                dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = SanaModelEmbedding(
                embedding_config.uuid, embedding_state, embedding_config.placeholder,
            )
            model.additional_embeddings.append(embedding)
            self._add_embedding_to_tokenizer(model.tokenizer, embedding.text_tokens)

    def _setup_embedding(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        model.embedding = None

        embedding_state = model.embedding_state
        if embedding_state is None:
            embedding_state = self._create_new_embedding(
                model.tokenizer,
                model.text_encoder,
                config.embedding.initial_embedding_text,
                config.embedding.token_count,
            )

        embedding_state = embedding_state.to(
            dtype=model.text_encoder.get_input_embeddings().weight.dtype,
            device=self.train_device,
        ).detach()

        model.embedding = SanaModelEmbedding(
            config.embedding.uuid, embedding_state, config.embedding.placeholder,
        )
        self._add_embedding_to_tokenizer(model.tokenizer, model.embedding.text_tokens)

    def _setup_embedding_wrapper(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer,
            orig_module=model.text_encoder.embed_tokens,
            additional_embeddings=[embedding.text_encoder_vector for embedding in model.additional_embeddings]
                                  + ([] if model.embedding is None else [model.embedding.text_encoder_vector]),
            additional_embedding_placeholders=[embedding.placeholder for embedding in model.additional_embeddings]
                                  + ([] if model.embedding is None else [model.embedding.placeholder]),
            additional_embedding_names=[embedding.uuid for embedding in model.additional_embeddings]
                                  + ([] if model.embedding is None else [model.embedding.uuid]),
        )
        model.embedding_wrapper.hook_to_module()

    def predict(
            self,
            model: SanaModel,
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

            is_align_prop_step = config.align_prop and (rand.random() < config.align_prop_probability)

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

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
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

            predicted_flow = model.transformer(
                latent_input.to(dtype=config.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=config.train_dtype.torch_dtype()),
                encoder_attention_mask=text_encoder_attention_mask.to(dtype=config.train_dtype.torch_dtype()),
                timestep=timestep,
            ).sample

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if self.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.tokenizer),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    if is_align_prop_step:
                        # noise
                        self._save_image(
                            self._project_latent_to_image(latent_noise),
                            config.debug_dir + "/training_batches",
                            "1-noise",
                            train_progress.global_step,
                        )

                        # image
                        self._save_image(
                            self._project_latent_to_image(scaled_latent_image),
                            config.debug_dir + "/training_batches",
                            "2-image",
                            model.train_progress.global_step,
                        )
                    else:
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
            model: SanaModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            betas=model.noise_scheduler.betas.to(device=self.train_device),
        ).mean()
