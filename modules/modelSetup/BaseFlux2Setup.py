from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.Flux2Model import Flux2Model
from modules.model.FluxModel import FluxModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_flux2_transformer,
    enable_checkpointing_for_mistral_encoder_layers,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseFlux2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "blocks": ["transformer_block"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_flux2_transformer(model.transformer, config)
            if model.text_encoder is not None:
                if model.is_dev():
                    model.text_encoder_offload_conductor = \
                        enable_checkpointing_for_mistral_encoder_layers(model.text_encoder, config)
                else:
                    model.text_encoder_offload_conductor = \
                        enable_checkpointing_for_qwen3_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding:
            raise NotImplementedError #TODO applies to Flux2?
#            apply_circular_padding_to_conv2d(model.vae)
#            apply_circular_padding_to_conv2d(model.transformer)
#            if model.transformer_lora is not None:
#                apply_circular_padding_to_conv2d(model.transformer_lora)

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
            model: Flux2Model,
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

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )
            latent_image = model.patchify_latents(batch['latent_image'].float())
            latent_height = latent_image.shape[-2]
            latent_width = latent_image.shape[-1]
            scaled_latent_image = model.scale_latents(latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(latent_height, latent_width)
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

            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.transformer.guidance_scale], device=self.train_device, dtype=model.train_dtype.torch_dtype())
                guidance = guidance.expand(latent_input.shape[0])
            else:
                guidance = None

            text_ids = model.prepare_text_ids(text_encoder_output)
            image_ids = model.prepare_latent_image_ids(latent_input)
            packed_latent_input = model.pack_latents(latent_input)

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                txt_ids=text_ids,
                img_ids=image_ids,
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
                #unpatchify, to make the shape match the mask shape of masked training:
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: Flux2Model,
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

    def prepare_text_caching(self, model: FluxModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
