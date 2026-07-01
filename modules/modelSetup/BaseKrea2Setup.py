from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.Krea2Model import Krea2Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_krea2_transformer,
    enable_checkpointing_for_qwen3vl_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers import Krea2Pipeline


class BaseKrea2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ff"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_krea2_transformer(model.transformer, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = \
                    enable_checkpointing_for_qwen3vl_encoder_layers(model.text_encoder, config)

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

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

    def predict(
            self,
            model: Krea2Model,
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
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
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

            # position ids: text tokens at origin, image tokens at latent-grid coords (patch_size = 2)
            text_seq_len = text_encoder_output.shape[1]
            grid_height = latent_input.shape[-2] // 2
            grid_width = latent_input.shape[-1] // 2
            position_ids = Krea2Pipeline.prepare_position_ids(
                text_seq_len, grid_height, grid_width, self.train_device
            )

            if torch.all(text_attention_mask):
                text_attention_mask = None

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                position_ids=position_ids,
                encoder_attention_mask=text_attention_mask,
                return_dict=False,
            )[0]

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
            model: Krea2Model,
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

    def prepare_text_caching(self, model: Krea2Model, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
