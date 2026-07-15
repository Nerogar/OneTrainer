from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_qwen3_encoder_layers,
    enable_checkpointing_for_qwen_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseAnimaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    # CosmosTransformerBlock has attn1 (self-attn), attn2 (cross-attn), ff (feedforward)
    LAYER_PRESETS = {
        "attn-mlp": ["attn1", "attn2", "ff"],
        "attn-only": ["attn1", "attn2"],
        "blocks": ["transformer_block"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_qwen_transformer, attention_mask=False)
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_qwen3_encoder_layers, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "vae", config.vae)

    def predict(
            self,
            model: AnimaModel,
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

            # Anima encode_text returns a plain Tensor (no mask); conditioner output is (B,512,1024).
            text_encoder_output = model.encode_text(
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
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # Anima latents are 5D (B,16,1,H/8,W/8) — no pack/unpack needed.
            # CosmosTransformer3DModel requires padding_mask in pixel space (1,1,H,W).
            latent_h, latent_w = scaled_noisy_latent_image.shape[-2], scaled_noisy_latent_image.shape[-1]
            padding_mask = scaled_noisy_latent_image.new_zeros(
                1, 1, latent_h * 8, latent_w * 8,
            ).to(dtype=model.train_dtype.torch_dtype())

            predicted_flow = model.transformer(
                hidden_states=scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

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
            model: AnimaModel,
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

    def prepare_text_caching(self, model: AnimaModel, config: TrainConfig):
        model.materialize_only("text_encoder")

        model.eval()
        torch_gc()
