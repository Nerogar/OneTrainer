from abc import ABCMeta
from random import Random

from modules.model.FluxModel import FluxModel, FluxModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.flux.FluxXFormersAttnProcessor import FluxXFormersAttnProcessor
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    create_checkpointed_forward,
    enable_checkpointing_for_clip_encoder_layers,
    enable_checkpointing_for_flux_transformer,
    enable_checkpointing_for_t5_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import set_nf4_compute_type
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.utils import is_xformers_available


class BaseFluxSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):

    def _setup_optimizations(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        if config.attention_mechanism == AttentionMechanism.DEFAULT:
            model.transformer.set_attn_processor(FluxAttnProcessor2_0())
        elif config.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                model.transformer.set_attn_processor(FluxXFormersAttnProcessor(model.train_dtype.torch_dtype()))
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif config.attention_mechanism == AttentionMechanism.SDP:
            model.transformer.set_attn_processor(FluxAttnProcessor2_0())

            if is_xformers_available():
                try:
                    model.vae.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )

        if config.gradient_checkpointing.enabled():
            enable_checkpointing_for_flux_transformer(
                model.transformer, self.train_device, self.temp_device, config.gradient_checkpointing.offload())
            if model.text_encoder_1 is not None:
                enable_checkpointing_for_clip_encoder_layers(
                    model.text_encoder_1, self.train_device, self.temp_device, config.gradient_checkpointing.offload())
            if model.text_encoder_2 is not None and config.train_text_encoder_2_or_embedding():
                enable_checkpointing_for_t5_encoder_layers(
                    model.text_encoder_2, self.train_device, self.temp_device, config.gradient_checkpointing.offload())

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

        set_nf4_compute_type(model.text_encoder_1, model.train_dtype)
        set_nf4_compute_type(model.text_encoder_2, model.text_encoder_2_train_dtype)
        set_nf4_compute_type(model.transformer, model.train_dtype)

    def _setup_additional_embeddings(
            self,
            model: FluxModel,
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

            embedding = FluxModelEmbedding(
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
            model: FluxModel,
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

        model.embedding = FluxModelEmbedding(
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
            model: FluxModel,
            config: TrainConfig,
    ):
        if model.tokenizer_1 is not None and model.text_encoder_1 is not None:
            model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer_1,
                orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
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
                orig_module=model.text_encoder_2.encoder.embed_tokens,
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
            model: FluxModel,
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

            is_align_prop_step = config.align_prop and (rand.random() < config.align_prop_probability)

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
            )

            latent_image = batch['latent_image']
            scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = \
                    (batch['latent_conditioning_image'] - vae_shift_factor) * vae_scaling_factor

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            if is_align_prop_step and not deterministic:
                dummy = torch.zeros((1,), device=self.train_device)
                dummy.requires_grad_(True)

                negative_text_encoder_output, negative_pooled_text_encoder_2_output = model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch['latent_image'].shape[0],
                    rand=rand,
                    text="",
                    text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                    text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                )
                negative_text_encoder_output = negative_text_encoder_output \
                    .expand((scaled_latent_image.shape[0], -1, -1))
                negative_pooled_text_encoder_2_output = negative_pooled_text_encoder_2_output \
                    .expand((scaled_latent_image.shape[0], -1))

                model.noise_scheduler.set_timesteps(config.align_prop_steps)

                scaled_noisy_latent_image = latent_noise

                timestep_high = int(config.align_prop_steps * config.max_noising_strength)
                timestep_low = \
                    int(config.align_prop_steps * config.max_noising_strength * (
                            1.0 - config.align_prop_truncate_steps))

                truncate_timestep_index = config.align_prop_steps - rand.randint(timestep_low, timestep_high)

                # original size of the image
                original_height = scaled_noisy_latent_image.shape[2] * 8
                original_width = scaled_noisy_latent_image.shape[3] * 8
                crops_coords_top = 0
                crops_coords_left = 0
                target_height = scaled_noisy_latent_image.shape[2] * 8
                target_width = scaled_noisy_latent_image.shape[3] * 8

                add_time_ids = torch.tensor([
                    original_height,
                    original_width,
                    crops_coords_top,
                    crops_coords_left,
                    target_height,
                    target_width
                ]).unsqueeze(0).expand((scaled_latent_image.shape[0], -1))

                add_time_ids = add_time_ids.to(
                    dtype=scaled_noisy_latent_image.dtype,
                    device=scaled_noisy_latent_image.device,
                )

                added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
                negative_added_cond_kwargs = {"text_embeds": negative_pooled_text_encoder_2_output,
                                              "time_ids": add_time_ids}

                checkpointed_unet = create_checkpointed_forward(model.unet, self.train_device, self.temp_device)

                for step in range(config.align_prop_steps):
                    timestep = model.noise_scheduler.timesteps[step] \
                        .expand((scaled_latent_image.shape[0],)) \
                        .to(device=model.unet.device)

                    if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                        latent_input = torch.concat(
                            [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                        )
                    else:
                        latent_input = scaled_noisy_latent_image

                    predicted_latent_noise = checkpointed_unet(
                        sample=latent_input,
                        timestep=timestep,
                        encoder_hidden_states=text_encoder_output,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    negative_predicted_latent_noise = checkpointed_unet(
                        sample=latent_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_text_encoder_output,
                        added_cond_kwargs=negative_added_cond_kwargs,
                    ).sample

                    cfg_grad = (predicted_latent_noise - negative_predicted_latent_noise)
                    cfg_predicted_latent_noise = negative_predicted_latent_noise + config.align_prop_cfg_scale * cfg_grad

                    scaled_noisy_latent_image = model.noise_scheduler \
                        .step(cfg_predicted_latent_noise, timestep[0].long(), scaled_noisy_latent_image) \
                        .prev_sample

                    if step < truncate_timestep_index:
                        scaled_noisy_latent_image = scaled_noisy_latent_image.detach()

                    if self.debug_mode:
                        with torch.no_grad():
                            # predicted image
                            predicted_image = self._project_latent_to_image(scaled_noisy_latent_image)
                            self._save_image(
                                predicted_image,
                                config.debug_dir + "/training_batches",
                                "2-predicted_image_" + str(step),
                                train_progress.global_step,
                                True
                            )

                predicted_latent_image = scaled_noisy_latent_image / vae_scaling_factor
                predicted_latent_image = predicted_latent_image.to(dtype=model.vae.dtype)

                predicted_image = []
                for x in predicted_latent_image.split(1):
                    predicted_image.append(torch.utils.checkpoint.checkpoint(
                        model.vae.decode,
                        x,
                        use_reentrant=False
                    ).sample)
                predicted_image = torch.cat(predicted_image)

                model_output_data = {
                    'loss_type': 'align_prop',
                    'predicted': predicted_image,
                }
            else:
                timestep_index = self._get_timestep_discrete(
                    model.noise_scheduler.config['num_train_timesteps'],
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    config,
                )

                scaled_noisy_latent_image, timestep, sigma = self._add_noise_discrete(
                    scaled_latent_image,
                    latent_noise,
                    timestep_index,
                    model.noise_scheduler.timesteps,
                )

                if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                    latent_input = torch.concat(
                        [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                    )
                else:
                    latent_input = scaled_noisy_latent_image

                if model.transformer.config.guidance_embeds:
                    guidance = torch.tensor([1.0], device=self.train_device)
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

        # model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
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
