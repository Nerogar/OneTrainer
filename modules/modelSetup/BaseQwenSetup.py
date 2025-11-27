from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.QwenModel import QwenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_qwen_encoder_layers,
    enable_checkpointing_for_qwen_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

PRESETS = {
    "attn-mlp": ["attn", "img_mlp", "txt_mlp"],
    "attn-only": ["attn"],
    "blocks": ["transformer_block"],
    "full": [],
}


#TODO share more code with other models
class BaseQwenSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_qwen_transformer(model.transformer, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = \
                    enable_checkpointing_for_qwen_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding: #TODO useful for Qwen?
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
        ], config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().text_encoder,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def predict(
            self,
            model: QwenModel,
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

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch['text_encoder_hidden_state'] \
                    if 'text_encoder_hidden_state' in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = model.scale_latents(latent_image)
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            latent_input = scaled_noisy_latent_image
            packed_latent_input = model.pack_latents(latent_input)

            #FIXME this is the only case that the transformer accepts:
            #see https://github.com/huggingface/diffusers/issues/12344
            #actual text sequence lengths can be shorter,but they might be padded and masked
            txt_seq_lens = [text_encoder_output.shape[1]] * text_encoder_output.shape[0]

            #FIXME list of lists is not according to type hint, but according to diffusers code:
            #https://github.com/huggingface/diffusers/issues/12295
            img_shapes = [[(
                1, #frame for future video model - not batch size
                latent_input.shape[-2] // 2,
                latent_input.shape[-1] // 2)
            ]] * latent_input.shape[0]

            #FIXME bug workaround for https://github.com/huggingface/diffusers/issues/12294
            image_attention_mask=torch.ones((packed_latent_input.shape[0], packed_latent_input.shape[1]), dtype=torch.bool, device=latent_image.device)
            attention_mask = torch.cat([text_attention_mask, image_attention_mask], dim=1)
            attention_mask_2d = attention_mask[:, None, None, :] if not torch.all(text_attention_mask) else None

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states_mask=text_attention_mask,
                txt_seq_lens=txt_seq_lens,
                img_shapes=img_shapes,
                attention_kwargs = {
                    "attention_mask": attention_mask_2d,
                },
                return_dict=True,
            ).sample

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                height=latent_input.shape[-2],
                width=latent_input.shape[-1],
            )

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode: #TODO simplify
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.tokenizer),
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
            model: QwenModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()
