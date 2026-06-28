from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.ErnieModel import ErnieModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_ernie_transformer,
    enable_checkpointing_for_mistral_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseErnieSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["self_attention", "mlp"],
        "attn-only": ["self_attention"],
        "blocks": ["layers"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: ErnieModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_ernie_transformer(model.transformer, config)
            model.text_encoder_offload_conductor = \
                enable_checkpointing_for_mistral_encoder_layers(model.text_encoder, config)

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
            model: ErnieModel,
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

            text_encoder_output, text_lens = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get('tokens'),
                tokens_mask=batch.get('tokens_mask'),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            # Patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]
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
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            predicted_flow = model.transformer(
                hidden_states=scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                text_bth=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                text_lens=text_lens,
                return_dict=False,
            )[0]

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                # unpatchify to match mask shape for masked training
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens('7-prompt', batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent('1-noise', latent_noise, config, train_progress)
                    self._save_latent('2-noisy_image', scaled_noisy_latent_image, config, train_progress)
                    self._save_latent('3-predicted_flow', predicted_flow, config, train_progress)
                    self._save_latent('4-flow', flow, config, train_progress)
                    self._save_latent('5-predicted_image', predicted_scaled_latent_image, config, train_progress)
                    self._save_latent('6-image', scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: ErnieModel,
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

    def prepare_text_caching(self, model: ErnieModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
