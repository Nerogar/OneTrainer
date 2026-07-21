from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.IdeogramModel import IdeogramModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_ideogram_transformer,
    enable_checkpointing_for_qwen3vl_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseIdeogramSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attention", "feed_forward"],
        "attn-only": ["attention"],
        "blocks": ["layers"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        # Only the conditional transformer is trained, so gradient checkpointing applies there.
        model.transformer_offload_conductor = \
            enable_checkpointing_for_ideogram_transformer(model.transformer, config, config.transformer)

        # The unconditional transformer is frozen, but it still benefits from layer offloading
        # since both transformers need to fit in VRAM during sampling. It is optional, so may be unloaded.
        if model.unconditional_transformer is not None:
            model.unconditional_transformer_offload_conductor = \
                enable_checkpointing_for_ideogram_transformer(model.unconditional_transformer, config, config.unconditional_transformer)

        model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3vl_encoder_layers(model.text_encoder, config, config.text_encoder)

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        quantize_layers(model.unconditional_transformer, self.train_device, model.train_dtype, config)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)
        if model.unconditional_transformer is not None:
            self._set_attention_backend(model.unconditional_transformer, config.attention_mechanism, mask=False)

    def predict(
            self,
            model: IdeogramModel,
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

            latent_image = batch['latent_image']  # (B, 32, H_lat, W_lat)
            batch_size = latent_image.shape[0]
            latent_height = latent_image.shape[-2]
            latent_width = latent_image.shape[-1]
            grid_h = latent_height // 2
            grid_w = latent_width // 2
            num_image_tokens = grid_h * grid_w

            text_encoder_output, text_lengths = model.encode_text(
                train_device=self.train_device,
                batch_size=batch_size,
                rand=rand,
                tokens=batch.get('tokens'),
                tokens_mask=batch.get('tokens_mask'),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )
            max_text_tokens = text_encoder_output.shape[1]

            # patchify [B, 32, H, W] -> packed (B, num_image_tokens, 128), then bn-normalize in packed space (the
            # sampler scales the packed sequence the same way).
            packed_latent_image = model.patchify_latents(latent_image.float())
            scaled_latent_image = model.scale_latents(packed_latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(latent_height, latent_width)
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                batch_size,
                config,
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # build the packed [left-pad][text][image] layout (shared helper; identical to the sampler)
            position_ids, segment_ids, indicator = model.prepare_packed_ids(
                text_lengths, grid_h, grid_w, max_text_tokens, self.train_device,
            )

            # encode_text already returns features left-aligned to this layout, so pack them directly
            dtype = model.train_dtype.torch_dtype()
            llm_features = model.pack_llm_features(text_encoder_output, num_image_tokens).to(dtype)

            # hidden states: zero latents over the text positions, the noisy image latents over the image positions
            text_z_padding = torch.zeros(
                batch_size, max_text_tokens, scaled_noisy_latent_image.shape[-1],
                dtype=torch.float32, device=self.train_device,
            )
            hidden_states = torch.cat([text_z_padding, scaled_noisy_latent_image], dim=1).to(dtype)

            # the transformer's timestep is flow-matching time in [0, 1] (0 = noise, 1 = data), not the discrete index;
            # sigma is the noise fraction from _add_noise_discrete, so model time = 1 - sigma (matches the sampler).
            model_time = (1.0 - sigma).reshape(batch_size)

            predicted = model.transformer(
                hidden_states=hidden_states,
                timestep=model_time,
                encoder_hidden_states=llm_features,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
                return_dict=False,
            )[0]
            predicted_flow = predicted[:, max_text_tokens:].float()

            # The transformer's velocity convention is data - noise.
            flow = scaled_latent_image - latent_noise
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                # unpatchify both back to (B, 32, H, W) to match the latent mask shape for masked training
                'predicted': model.unpatchify_latents(predicted_flow, grid_h, grid_w),
                'target': model.unpatchify_latents(flow, grid_h, grid_w),
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image + predicted_flow * sigma
                    self._save_tokens('7-prompt', batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent('1-noise', model.unpatchify_latents(latent_noise, grid_h, grid_w), config, train_progress)
                    self._save_latent('2-noisy_image', model.unpatchify_latents(scaled_noisy_latent_image, grid_h, grid_w), config, train_progress)
                    self._save_latent('3-predicted_flow', model.unpatchify_latents(predicted_flow, grid_h, grid_w), config, train_progress)
                    self._save_latent('4-flow', model.unpatchify_latents(flow, grid_h, grid_w), config, train_progress)
                    self._save_latent('5-predicted_image', model.unpatchify_latents(predicted_scaled_latent_image, grid_h, grid_w), config, train_progress)
                    self._save_latent('6-image', model.unpatchify_latents(scaled_latent_image, grid_h, grid_w), config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: IdeogramModel,
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

    def prepare_text_caching(self, model: IdeogramModel, config: TrainConfig):
        model.materialize_only("text_encoder")
        model.eval()
