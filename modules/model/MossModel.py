import math
from contextlib import nullcontext
from random import Random

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import AutoencoderKLFlux2, DiffusionPipeline
from transformers import Qwen2Tokenizer, Qwen3_5Model

from modules.model.BaseModel import BaseModel
from modules.model.moss.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
)
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType

# Maximum number of tokens for a chat-templated prompt (template overhead + user content).
PROMPT_MAX_LENGTH = 4096


def format_moss_prompt(tokenizer: Qwen2Tokenizer, prompt: str) -> str:
    """Render a caption through the MOSS chat template (matches the official pipeline's encode_prompt)."""
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


class MossModelEmbedding:
    def __init__(
            self,
            uuid: str,
            placeholder: str,
            token_count: int,
    ):
        self.uuid = uuid
        self.placeholder = placeholder
        self.token_count = token_count
        self.adjoint_token: Tensor | None = None


class MossModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: MOSSImageScheduler | None
    text_encoder: Qwen3_5Model | None
    vae: AutoencoderKLFlux2 | None
    transformer: MOSSImageTransformer | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    # persistent lora training data
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    embedding: MossModelEmbedding | None
    additional_embeddings: list[MossModelEmbedding] | None
    adjoint_tokens: dict[str, Tensor] | None

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

        self.text_encoder_autocast_context = nullcontext()

        self.transformer_lora = None
        self.lora_state_dict = None

        self.embedding = None
        self.additional_embeddings = []
        self.adjoint_tokens = {}

    def diffusers_to_original(self) -> list | None:
        # MOSS's native checkpoint is a strict rename-free mapping of the MOSSImageTransformer state
        # dict (see MOSSImageTransformer.load_training_state_dict), so the diffusers and original
        # formats are identical.
        return None

    def create_pipeline(self) -> DiffusionPipeline:
        return MOSSImagePipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
            apply_adjoint_tokens: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokens_out = self.tokenizer(
                [format_moss_prompt(self.tokenizer, t) for t in text],
                truncation=True,
                padding='max_length',
                max_length=PROMPT_MAX_LENGTH,
                return_tensors='pt',
            )
            tokens = tokens_out.input_ids.to(self.text_encoder.device)
            tokens_mask = tokens_out.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is not None:
            text_encoder_output = text_encoder_output.to(device=train_device)
        if tokens_mask is not None:
            tokens_mask = tokens_mask.to(device=train_device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    output_hidden_states=True,
                )
                # single last hidden state → (B, T, H)
                text_encoder_output = outputs.hidden_states[-1]

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # real tokens are a contiguous prefix (padding sits at the end), so gather the
        # unmasked tokens per sample, append adjoint tokens, then re-pad the batch
        seq_lengths = tokens_mask.sum(dim=1)
        pruned_hidden_states = [text_encoder_output[i][tokens_mask[i].bool()] for i in range(text_encoder_output.shape[0])]

        if apply_adjoint_tokens and self.adjoint_tokens:
            active_adjoint_tokens = [
                self.adjoint_tokens[emb.uuid]
                for emb in self.all_embeddings()
                if emb.uuid in self.adjoint_tokens
            ]
            if active_adjoint_tokens:
                adjoint_concat = torch.cat(active_adjoint_tokens, dim=0).to(
                    device=train_device, dtype=text_encoder_output.dtype
                )
                pruned_hidden_states = [
                    torch.cat([h, adjoint_concat], dim=0) for h in pruned_hidden_states
                ]
                seq_lengths = seq_lengths + adjoint_concat.shape[0]

        text_encoder_output = torch.nn.utils.rnn.pad_sequence(pruned_hidden_states, batch_first=True)
        max_seq_length = text_encoder_output.shape[1]

        # pad to a multiple of 16 to keep cuDNN's fused-attention shape cache hot across steps
        if max_seq_length % 16 != 0 and (seq_lengths != seq_lengths[0]).any():
            padded_length = max_seq_length + (16 - max_seq_length % 16)
            text_encoder_output = F.pad(text_encoder_output, (0, 0, 0, padded_length - max_seq_length))
            max_seq_length = padded_length

        bool_attention_mask = torch.arange(max_seq_length, device=train_device).unsqueeze(0) < seq_lengths.unsqueeze(1)

        return (text_encoder_output, bool_attention_mask)

    def scale_latents(self, latents: Tensor) -> Tensor:
        # the FLUX.2 VAE emits 32-channel latents at 1/8 image scale, but the transformer
        # consumes a 2x2-packed 128-channel latent at 1/16 scale, normalized with the VAE's
        # batch-norm running statistics (its bn has 128 channels - one per packed channel).
        # Pack the 2x2 spatial blocks into 4x channels, then normalize in packed space.
        b, c, h, w = latents.shape
        packed = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        return (packed - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        # reverse of scale_latents: denormalize in packed 128-channel space, then unpack
        # the 2x2 blocks back to 32 channels at 2x spatial resolution for the VAE decoder.
        b, c, h, w = latents.shape
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        denormed = latents * std + mean
        return denormed.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)

    def all_embeddings(self) -> list[MossModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def _apply_adjoint_tokens(self, text_encoder_output: Tensor) -> Tensor:
        # Adjoint tokens are concatenated directly in encode_text to preserve real token lengths
        return text_encoder_output

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift

        # the MOSS transformer consumes the VAE latents at full spatial resolution
        # (VAE scale factor 16, 128 channels), so the image sequence length is the
        # latent grid size as-is.
        image_seq_len = latent_width * latent_height
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
