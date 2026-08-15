from abc import ABCMeta

import modules.util.multi_gpu_util as multi
from modules.model.LTXModel import LTXModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_gemma3_encoder_layers,
    enable_checkpointing_for_gemma4_encoder_layers,
    enable_checkpointing_for_ltx_connectors,
    enable_checkpointing_for_ltx_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from transformers import Gemma3ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration


class BaseLTXSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    # The leading dot keeps LoRA on the video submodules - the audio counterparts are "audio_attn1" etc.
    #
    # The "lightricks-*" entries are quantization filters, not LoRA target sets (the two dropdowns share this
    # dict). Each reproduces one Lightricks prequantized release, read off that checkpoint's safetensors
    # header. Spelled as exclusions because regex has no numeric ranges, so they encode the 48-block count;
    # "to_gate_logits" needs the ".*" because a Linear directly in a ModuleList is filtered under its parent's
    # name. The dict form with regex=True is how a preset carries the "Use Regex" toggle.
    LAYER_PRESETS = {
        "video-attn-mlp": [".attn1", ".attn2", ".ff"],
        "video-attn-only": [".attn1", ".attn2"],
        "blocks": ["transformer_blocks"],
        "lightricks-2.3-fp8": {"patterns": [r"^transformer_blocks\.(?![01]\.|4[67]\.)"], "regex": True},
        "lightricks-2.5-int8-convrot": {
            "patterns": [r"^transformer_blocks\.\d+\.(?!.*to_gate_logits)"], "regex": True},
        "lightricks-2.5-nvfp4": {
            "patterns": [r"^transformer_blocks\.(?!4[2-7]\.)\d+\.(?!.*to_gate_logits)"], "regex": True},
        "full": [],
    }

    def setup_optimizations(
            self,
            model: LTXModel,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_ltx_transformer, disable_fp16_autocast=True)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)
        if isinstance(model.text_encoder, Gemma3ForConditionalGeneration):
            enable_checkpointing_for_text_encoder = enable_checkpointing_for_gemma3_encoder_layers
        elif isinstance(model.text_encoder, Gemma4UnifiedForConditionalGeneration):
            enable_checkpointing_for_text_encoder = enable_checkpointing_for_gemma4_encoder_layers
        else:
            raise NotImplementedError(f"no checkpointing wrapper for text encoder {type(model.text_encoder)}")
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_text_encoder, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "connectors", config.connectors, enable_checkpointing_for_ltx_connectors, disable_fp16_autocast=True)
        if model.low_noise_transformer is not None:
            self._setup_model_part(model, config, "low_noise_transformer", config.low_noise_transformer, enable_checkpointing_for_ltx_transformer, disable_fp16_autocast=True)
            self._set_attention_backend(model.low_noise_transformer, config.attention_mechanism, mask=True)
        self._setup_model_part(model, config, "vae", config.vae)

        model.vae.enable_tiling()

    def predict(
            self,
            model: LTXModel,
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

            # only the video conditioning is cached: the audio branch is isolated (isolate_modalities below)
            # and discarded, so zeros of the right shape are fed for it instead
            connector_prompt_embeds = model.encode_text(
                train_device=self.train_device,
                connector_video_embeds=batch['connector_video_embeds'],
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            latent_image = batch['latent_image'].float()
            # a video sample carries a frame dim, an image sample does not, and a dataset can hold both
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)
            batch_size, _, num_latent_frames, latent_height, latent_width = latent_image.shape

            scaled_latent_image = model.scale_latents(latent_image)
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(num_latent_frames, latent_height, latent_width)
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

            patch_size = model.transformer.config.patch_size
            patch_size_t = model.transformer.config.patch_size_t
            packed_noisy_latent_image = model.pack_latents(scaled_noisy_latent_image, patch_size, patch_size_t)

            # video-only scope: isolate_modalities=True skips the cross-modality attention blocks, so the audio
            # inputs the transformer takes positionally only need the right shape - hence zeros. Without it the
            # video LoRA would take gradients from nonsense audio content every step.
            frame_rate = model.NATIVE_FPS
            pixel_num_frames = (num_latent_frames - 1) * model.vae.temporal_compression_ratio + 1
            duration_s = pixel_num_frames / frame_rate
            audio_latents_per_second = (
                model.audio_vae.config.sample_rate
                / model.audio_vae.config.mel_hop_length
                / float(model.audio_vae.temporal_compression_ratio)
            )
            audio_num_frames = round(duration_s * audio_latents_per_second)
            latent_mel_bins = model.audio_vae.config.mel_bins // model.audio_vae.mel_compression_ratio
            audio_shape = (batch_size, audio_num_frames, model.audio_vae.config.latent_channels * latent_mel_bins)
            with model.transformer_autocast_context:
                dtype = model.transformer_train_dtype.torch_dtype()
                predicted_flow, _ = model.transformer(
                    hidden_states=packed_noisy_latent_image.to(dtype=dtype),
                    audio_hidden_states=torch.zeros(audio_shape, device=self.train_device, dtype=dtype),
                    encoder_hidden_states=connector_prompt_embeds.to(dtype=dtype),
                    audio_encoder_hidden_states=torch.zeros(
                        (batch_size, connector_prompt_embeds.shape[1], model.connectors.config.audio_hidden_dim),
                        device=self.train_device, dtype=dtype,
                    ),
                    timestep=timestep,
                    sigma=timestep,
                    encoder_attention_mask=None,
                    audio_encoder_attention_mask=None,
                    num_frames=num_latent_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    isolate_modalities=True,
                    return_dict=False,
                )

            # unpack, to make the shape match the mask shape of masked training. LTX's unpack also unfolds the
            # patches, so it lands on [B, C, F, H, W] in one step
            predicted_flow = model.unpack_latents(
                predicted_flow, num_latent_frames, latent_height, latent_width, patch_size, patch_size_t,
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
            model: LTXModel,
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

    def prepare_text_caching(self, model: LTXModel, config: TrainConfig):
        # the connector output is what gets cached, so the connectors run in the caching pass too
        model.materialize_only("text_encoder", "connectors")
        model.eval()
