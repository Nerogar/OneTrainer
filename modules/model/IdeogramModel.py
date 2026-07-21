import math
from contextlib import nullcontext
from random import Random

PROMPT_MAX_LENGTH = 2048

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    Ideogram4Pipeline,
    Ideogram4Transformer2DModel,
)
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import _resolution_aware_mu
from transformers import Qwen2TokenizerFast, Qwen3VLModel


class IdeogramModel(BaseModel):
    # base model data
    tokenizer: Qwen2TokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLModel | None
    vae: AutoencoderKLFlux2 | None
    transformer: Ideogram4Transformer2DModel | None
    # second, frozen DiT for the asymmetric (dual-network) classifier-free guidance — only used at sampling
    unconditional_transformer: Ideogram4Transformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None
    unconditional_transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.unconditional_transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None
        self.unconditional_transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def fusion_groups(self) -> list | None:
        # Ideogram4 fuses q/k/v into one qkv Linear per block; everything else in the transformer -- including
        # the output projection (to_out.0 -> o, see diffusers_to_original) -- already matches the original
        # checkpoint's naming.
        return [
            ("layers.{i}", ["attention.to_q", "attention.to_k", "attention.to_v"], "attention.qkv", "attention.qkv"),
        ]

    def diffusers_to_original(self) -> list | None:
        # Ideogram4's native (ComfyUI/original) checkpoint is identical to diffusers except for one rename:
        # to_out.0 -> o on the (already fused, see fusion_groups above) attention module. Everything else --
        # feed_forward, norms, adaln_modulation, final_layer, t_embedding, etc. -- keeps its diffusers name, so
        # a single catch-all identity pattern covers the rest.
        return [
            ("layers.{i}.attention.to_out.0", "layers.{i}.attention.o"),
            ("{path}", "{path}"),
        ]

    def create_pipeline(self) -> DiffusionPipeline:
        return Ideogram4Pipeline(
            transformer=self.transformer,
            unconditional_transformer=self.unconditional_transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
            prompt_enhancer_head=None,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            # Ideogram wraps each prompt in a chat template, like Z-Image. Source: Ideogram4Pipeline.encode_prompt.
            for i, prompt_item in enumerate(text):
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_item}]}]
                text[i] = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            tokenizer_output = self.tokenizer(
                text,
                padding='max_length',
                max_length=PROMPT_MAX_LENGTH,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False,
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                # Mirrors Ideogram4Pipeline.encode_prompt: text-only MRoPE shares the linear token position across
                # all 3 axes, so a plain arange is the position_ids for the real tokens.
                position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
                selected = Ideogram4Pipeline._get_text_encoder_hidden_states(self.text_encoder, tokens, tokens_mask, position_ids)

                # Interleave by hidden dim (NOT torch.cat): stack -> (L, B, T, H), permute -> (B, T, H, L), reshape ->
                # (B, T, H*L). This grouped-by-hidden-dim order is what the transformer's llm_cond_proj expects.
                text_encoder_output = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(tokens.shape[0], tokens.shape[1], -1)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # needs empty-caption conditioning, not zero-out

        # Run unconditionally in both the fresh and cached paths: zero out padding positions.
        text_encoder_output = text_encoder_output * tokens_mask.to(text_encoder_output.dtype).unsqueeze(-1)

        text_lengths = tokens_mask.sum(dim=1).long()

        # Left-align each sample's real tokens to the END of the text block, matching the [left-pad][text][image]
        # layout that prepare_packed_ids (Ideogram4Pipeline._prepare_ids) builds. The transformer masks padding via
        # segment_ids, so the padding's sequence position is otherwise irrelevant; aligning to the ids' convention here
        # lets every caller pack the features directly, with no second alignment pass (the single-prompt sampler has
        # offset 0, so this is a no-op there). Real tokens are front-aligned coming in (right-padded tokenizer output).
        max_text_tokens = int(text_lengths.max().item())
        aligned_output = text_encoder_output.new_zeros(
            text_encoder_output.shape[0], max_text_tokens, text_encoder_output.shape[-1],
        )
        for b, num_text in enumerate(text_lengths.tolist()):
            aligned_output[b, max_text_tokens - num_text:] = text_encoder_output[b, :num_text]
        return aligned_output, text_lengths

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) VAE latents -> packed (B, L, C * 4) sequence with L = (H // 2) * (W // 2).
        # Inverse of the unpatchify in Ideogram4Pipeline.__call__'s decode tail.
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 3, 5, 1)
        return latents.reshape(b, (h // 2) * (w // 2), c * 4)

    @staticmethod
    def prepare_packed_ids(
            text_lengths: list[int] | Tensor,
            grid_h: int,
            grid_w: int,
            max_text_tokens: int,
            device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Ideogram4Pipeline._prepare_ids takes a plain list[int]; encode_text returns text_lengths as a Tensor.
        if isinstance(text_lengths, Tensor):
            text_lengths = text_lengths.tolist()
        return Ideogram4Pipeline._prepare_ids(text_lengths, grid_h, grid_w, max_text_tokens, device)

    @staticmethod
    def pack_llm_features(text_features: Tensor, num_image_tokens: int) -> Tensor:
        # Assemble the packed [text][image] conditioning: the image positions carry zeroed text features (they are
        # conditioned via position_ids/segment_ids/indicator, not features). Mirrors Ideogram4Pipeline.encode_prompt.
        # The caller supplies text_features already aligned to the target layout (front-aligned for the single-prompt
        # sampler, left-padded for the batched training predict).
        image_feature_padding = torch.zeros(
            text_features.shape[0], num_image_tokens, text_features.shape[-1],
            dtype=text_features.dtype, device=text_features.device,
        )
        return torch.cat([text_features, image_feature_padding], dim=1)

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        # packed (B, L, C * 4) -> (B, C, H, W). Mirrors Ideogram4Pipeline.__call__'s decode unpatchify.
        b, _l, cp = latents.shape
        c = cp // 4
        latents = latents.view(b, grid_h, grid_w, 2, 2, c)
        latents = latents.permute(0, 5, 1, 3, 2, 4)
        return latents.reshape(b, c, grid_h * 2, grid_w * 2)

    def scale_latents(self, latents: Tensor) -> Tensor:
        # Operates on packed (B, L, C * 4); vae.bn stats are per packed-channel, so (C*4,) broadcasts on the last dim.
        # Same batch-norm de/normalization as Flux2Model.scale_latents (same AutoencoderKLFlux2), duplicated here only
        # because that copy uses the unpacked (1, -1, 1, 1) broadcast; matches Ideogram4Pipeline.__call__'s bn scaling.
        mean = self.vae.bn.running_mean.view(1, 1, -1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            self.vae.bn.running_var.view(1, 1, -1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        mean = self.vae.bn.running_mean.view(1, 1, -1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            self.vae.bn.running_var.view(1, 1, -1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * std + mean

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        # Ideogram shifts the flow-matching schedule by a resolution-aware mu (pipeline _resolution_aware_mu, relative to
        # a 512x512 base). OneTrainer's flow-matching timestep sampling applies a multiplicative shift, which is exp(mu)
        # (same mu->shift mapping as Ernie). _resolution_aware_mu only depends on the pixel-count ratio, so convert the
        # latent dims back to pixels (VAE scale factor 8).
        mu = _resolution_aware_mu(height=latent_height * 8, width=latent_width * 8, base_mu=0.0)
        return math.exp(mu)
