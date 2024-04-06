from abc import ABCMeta
from random import Random

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionNoiseMixin import ModelSetupDiffusionNoiseMixin
from modules.modelSetup.stableDiffusion.checkpointing_util import \
    enable_checkpointing_for_transformer_blocks, enable_checkpointing_for_clip_encoder_layers, \
    create_checkpointed_forward
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseStableDiffusionXLSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):

    def _setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.attention_mechanism == AttentionMechanism.DEFAULT:
            model.unet.set_attn_processor(AttnProcessor())
        elif config.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                model.unet.set_attn_processor(XFormersAttnProcessor())
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif config.attention_mechanism == AttentionMechanism.SDP:
            model.unet.set_attn_processor(AttnProcessor2_0())

            if is_xformers_available():
                try:
                    model.vae.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )

        if config.gradient_checkpointing:
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_transformer_blocks(model.unet, self.train_device)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, self.train_device)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, self.train_device)

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

    def _setup_additional_embeddings(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.additional_embeddings = []
        for i, embedding_config in enumerate(config.additional_embeddings):
            embedding_state = model.additional_embedding_states[i]
            if embedding_state is None:
                embedding_state_1 = self._create_new_embedding(
                    model.tokenizer_1,
                    model.text_encoder_1,
                    config.additional_embeddings[i].initial_embedding_text,
                    config.additional_embeddings[i].token_count,
                )

                embedding_state_2 = self._create_new_embedding(
                    model.tokenizer_2,
                    model.text_encoder_2,
                    config.additional_embeddings[i].initial_embedding_text,
                    config.additional_embeddings[i].token_count,
                )
            else:
                embedding_state_1, embedding_state_2 = embedding_state

            embedding_state_1 = embedding_state_1.to(
                dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding_state_2 = embedding_state_2.to(
                dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionXLModelEmbedding(
                embedding_config.uuid, embedding_state_1, embedding_state_2, embedding_config.placeholder,
            )
            model.additional_embeddings.append(embedding)
            self._add_embedding_to_tokenizer(model.tokenizer_1, embedding.text_tokens)
            self._add_embedding_to_tokenizer(model.tokenizer_2, embedding.text_tokens)


    def _setup_embedding(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.embedding = None

        embedding_state = model.embedding_state
        if embedding_state is None:
            embedding_state_1 = self._create_new_embedding(
                model.tokenizer_1,
                model.text_encoder_1,
                config.embedding.initial_embedding_text,
                config.embedding.token_count,
            )

            embedding_state_2 = self._create_new_embedding(
                model.tokenizer_2,
                model.text_encoder_2,
                config.embedding.initial_embedding_text,
                config.embedding.token_count,
            )
        else:
            embedding_state_1, embedding_state_2 = embedding_state

        embedding_state_1 = embedding_state_1.to(
            dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
            device=self.train_device,
        ).detach()

        embedding_state_2 = embedding_state_2.to(
            dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
            device=self.train_device,
        ).detach()

        model.embedding = StableDiffusionXLModelEmbedding(
            config.embedding.uuid, embedding_state_1, embedding_state_2, config.embedding.placeholder,
        )
        self._add_embedding_to_tokenizer(model.tokenizer_1, model.embedding.text_tokens)
        self._add_embedding_to_tokenizer(model.tokenizer_2, model.embedding.text_tokens)

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_1,
            orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
            additional_embeddings=[embedding.text_encoder_1_vector for embedding in model.additional_embeddings]
                                  + ([] if model.embedding is None else [model.embedding.text_encoder_1_vector]),
        )
        model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_2,
            orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
            additional_embeddings=[embedding.text_encoder_2_vector for embedding in model.additional_embeddings]
                                  + ([] if model.embedding is None else [model.embedding.text_encoder_2_vector]),
        )

        model.embedding_wrapper_1.hook_to_module()
        model.embedding_wrapper_2.hook_to_module()

    def __encode_text(
            self,
            model: StableDiffusionXLModel,
            text_encoder_layer_skip: int,
            text_encoder_2_layer_skip: int,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
            text: str = None,
    ):
        if tokens_1 is None and text is not None:
            tokenizer_output = model.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(model.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokenizer_output = model.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(model.text_encoder_2.device)

        if text_encoder_1_output is None:
            text_encoder_1_output = model.text_encoder_1(
                tokens_1, output_hidden_states=True, return_dict=True
            )
            text_encoder_1_output = text_encoder_1_output.hidden_states[-(2 + text_encoder_layer_skip)]

        if text_encoder_2_output is None or pooled_text_encoder_2_output is None:
            text_encoder_2_output = model.text_encoder_2(
                tokens_2, output_hidden_states=True, return_dict=True
            )
            pooled_text_encoder_2_output = text_encoder_2_output.text_embeds
            text_encoder_2_output = text_encoder_2_output.hidden_states[-(2 + text_encoder_2_layer_skip)]

        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output

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
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(train_progress.global_step)
            rand = Random(train_progress.global_step)

            is_align_prop_step = config.align_prop and (rand.random() < config.align_prop_probability)

            vae_scaling_factor = model.vae.config['scaling_factor']

            text_encoder_output, pooled_text_encoder_2_output = self.__encode_text(
                model,
                config.text_encoder_layer_skip,
                config.text_encoder_2_layer_skip,
                tokens_1=batch['tokens_1'],
                tokens_2=batch['tokens_2'],
                text_encoder_1_output=batch[
                    'text_encoder_1_hidden_state'] if not config.text_encoder.train and not config.train_any_embedding() else None,
                text_encoder_2_output=batch[
                    'text_encoder_2_hidden_state'] if not config.text_encoder_2.train and not config.train_any_embedding() else None,
                pooled_text_encoder_2_output=batch[
                    'text_encoder_2_pooled_state'] if not config.text_encoder_2.train and not config.train_any_embedding() else None,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            if is_align_prop_step and not deterministic:
                dummy = torch.zeros((1,), device=self.train_device)
                dummy.requires_grad_(True)

                negative_text_encoder_output, negative_pooled_text_encoder_2_output = self.__encode_text(
                    model,
                    config.text_encoder_layer_skip,
                    config.text_encoder_2_layer_skip,
                    text="",
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

                checkpointed_unet = create_checkpointed_forward(model.unet, self.train_device)

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
                            predicted_image = self._project_latent_to_image_sdxl(scaled_noisy_latent_image)
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
                timestep = self._get_timestep_discrete(
                    model.noise_scheduler,
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    config,
                    train_progress.global_step,
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
                    sample=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=text_encoder_output,
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

                    if is_align_prop_step:
                        # noise
                        self._save_image(
                            self._project_latent_to_image_sdxl(latent_noise),
                            config.debug_dir + "/training_batches",
                            "1-noise",
                            train_progress.global_step,
                            True
                        )

                        # image
                        self._save_image(
                            self._project_latent_to_image_sdxl(scaled_latent_image),
                            config.debug_dir + "/training_batches",
                            "2-image",
                            model.train_progress.global_step,
                            True
                        )
                    else:
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
            betas=model.noise_scheduler.betas.to(device=self.train_device),
        ).mean()
