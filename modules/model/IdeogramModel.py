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

    def adapters(self) -> list[LoRAModuleWrapper]:
        # only the conditional transformer is trainable; the unconditional transformer never sees the concept
        return [a for a in [
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def unconditional_transformer_to(self, device: torch.device):
        if self.unconditional_transformer is not None:
            if self.unconditional_transformer_offload_conductor is not None and \
                    self.unconditional_transformer_offload_conductor.layer_offload_activated():
                self.unconditional_transformer_offload_conductor.to(device)
            else:
                self.unconditional_transformer.to(device=device)

    def release(self):
        temp_device = torch.device(self.train_config.temp_device)
        self.vae_to(temp_device)
        self.text_encoder_to(temp_device)
        self.transformer_to(temp_device)
        self.unconditional_transformer_to(temp_device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()
        if self.unconditional_transformer is not None:
            self.unconditional_transformer.eval()

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

        if text_encoder_output is None and self.text_encoder is not None:
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
        text_encoder_output = text_encoder_output[:, :text_lengths.max().item(), :]
        return text_encoder_output, text_lengths

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
    def unpatchify_latents(latents: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        # packed (B, L, C * 4) -> (B, C, H, W). Mirrors Ideogram4Pipeline.__call__'s decode unpatchify.
        b, _l, cp = latents.shape
        c = cp // 4
        latents = latents.view(b, grid_h, grid_w, 2, 2, c)
        latents = latents.permute(0, 5, 1, 3, 2, 4)
        return latents.reshape(b, c, grid_h * 2, grid_w * 2)

    def scale_latents(self, latents: Tensor) -> Tensor:
        # Operates on packed (B, L, C * 4); vae.bn stats are per packed-channel, so (C*4,) broadcasts on the last dim.
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
