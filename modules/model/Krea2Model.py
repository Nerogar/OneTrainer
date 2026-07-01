import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import (
    AutoencoderKLQwenImage,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3VLModel

# Selected decoder-layer indices whose hidden states are stacked per token and fed to the transformer.
# Verified from Krea2Pipeline defaults: (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Chat-template prefix that is prepended to every prompt before tokenization. Uses the same
# str.format() convention as QwenModel.DEFAULT_PROMPT_TEMPLATE, so this string is reusable
# as-is for the DataLoader's Tokenize node (format_text=PROMPT_TEMPLATE_PREFIX).
PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}"
)
# Suffix appended after the padded prompt block (tokenized separately and concatenated).
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
# Number of prefix tokens to crop from the encoder output (= len(tokenize(PREFIX))).
PROMPT_TEMPLATE_CROP_START = 34
# Number of tokens in the suffix.
PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS = 5
# User-visible caption budget (not counting prefix or suffix).
PROMPT_MAX_LENGTH = 512


class Krea2Model(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLModel | None
    vae: AutoencoderKLQwenImage | None
    transformer: Krea2Transformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
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

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def diffusers_to_original(self) -> list | None:
        # Krea 2's native checkpoint (krea/Krea-2-Raw's raw.safetensors) is a pure rename of the diffusers
        # Krea2Transformer2DModel state dict -- q/k/v are already split in both namespaces, so no qkv fusion
        # stage is needed (fusion_groups() stays None). The per-block modulation table is reshaped, not just
        # renamed: native stores it flattened ([36864]), diffusers as a [6, 6144] table.
        def flatten_mod(t): return t.reshape(-1)
        def table_mod(t): return t.reshape(6, -1)

        block_body = [
            ("attn.to_gate",       "attn.gate"),
            ("attn.norm_k.weight", "attn.qknorm.knorm.scale"),
            ("attn.norm_q.weight", "attn.qknorm.qnorm.scale"),
            ("attn.to_k",          "attn.wk"),
            ("attn.to_out.0",      "attn.wo"),
            ("attn.to_q",          "attn.wq"),
            ("attn.to_v",          "attn.wv"),
            ("ff.down",            "mlp.down"),
            ("ff.gate",            "mlp.gate"),
            ("ff.up",              "mlp.up"),
            ("norm1.weight",       "prenorm.scale"),
            ("norm2.weight",       "postnorm.scale"),
        ]
        return [
            ("img_in",                    "first"),
            ("final_layer.linear",        "last.linear"),
            ("final_layer.norm.weight",   "last.norm.scale"),
            ("final_layer.scale_shift_table", "last.modulation.lin"),
            ("time_embed.linear_1",       "tmlp.0"),
            ("time_embed.linear_2",       "tmlp.2"),
            ("time_mod_proj",             "tproj.1"),
            ("txt_in.norm.weight",        "txtmlp.0.scale"),
            ("txt_in.linear_1",           "txtmlp.1"),
            ("txt_in.linear_2",           "txtmlp.3"),
            ("text_fusion.projector",     "txtfusion.projector"),
            ("text_fusion.layerwise_blocks.{i}", "txtfusion.layerwise_blocks.{i}", block_body),
            ("text_fusion.refiner_blocks.{i}",   "txtfusion.refiner_blocks.{i}",   block_body),
            ("transformer_blocks.{i}",    "blocks.{i}", [*block_body, ("scale_shift_table", "mod.lin", flatten_mod, table_mod)]),
        ]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device): #TODO share more code between models
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return Krea2Pipeline(
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
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            # tokenize prefix+prompt to a fixed length (suffix appended separately so it sits
            # after the padding, matching the layout the model was trained with)
            prefix_out = self.tokenizer(
                [PROMPT_TEMPLATE_PREFIX.format(t) for t in text],
                truncation=True,
                padding='max_length',
                max_length=PROMPT_MAX_LENGTH + PROMPT_TEMPLATE_CROP_START - PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
                return_tensors="pt",
            )
            suffix_out = self.tokenizer(
                [PROMPT_TEMPLATE_SUFFIX] * len(text),
                return_tensors="pt",
            )
            tokens = torch.cat([prefix_out.input_ids, suffix_out.input_ids], dim=1).to(self.text_encoder.device)
            tokens_mask = torch.cat([prefix_out.attention_mask, suffix_out.attention_mask], dim=1).bool().to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                # cumsum position_ids: suffix tokens get positions right after real prompt tokens,
                # not after the padding block — matches how the model was sampled at training time.
                position_ids = (tokens_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                outputs = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )
                # stack selected layers → (B, T, L, H); crop prefix; flatten to (B, T, L*H) so
                # the shape matches the DataLoader's PruneMaskedTokens/PadMaskedTokens cache
                # convention (unflattened back to (B, T, L, H) below, after both branches).
                hidden_states = torch.stack(
                    [outputs.hidden_states[i] for i in TEXT_ENCODER_SELECT_LAYERS], dim=2
                )
                hidden_states = hidden_states[:, PROMPT_TEMPLATE_CROP_START:]
                tokens_mask = tokens_mask[:, PROMPT_TEMPLATE_CROP_START:]
                text_encoder_output = hidden_states.flatten(2, 3)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # NOTE: unlike Qwen, Krea2's real tokens are not a contiguous prefix of the sequence -
        # padding sits between the (variable-length) prompt and the fixed suffix
        # ("<|im_end|>\n<|im_start|>assistant\n"). Gather only the real (mask==True) tokens per
        # sample - same approach as mgds's PruneMaskedTokens - then pad each back to this
        # batch's real-token max length.
        seq_lengths = tokens_mask.sum(dim=1)
        pruned_hidden_states = [text_encoder_output[i][tokens_mask[i].bool()] for i in range(text_encoder_output.shape[0])]

        text_encoder_output = torch.nn.utils.rnn.pad_sequence(pruned_hidden_states, batch_first=True)
        max_seq_length = text_encoder_output.shape[1]

        # round up to a multiple of 16 to reduce torch.compile recompiles from ever-changing
        # sequence lengths, but only if an attention mask is needed anyway - i.e. skip when every
        # sample has the same real length (always true at batch size 1). This also matters below
        # the traced graph: cuDNN's fused attention builds/caches an execution plan per exact
        # tensor shape, and a first-time shape costs ~30x a cached repeat (measured: ~30-70ms cold
        # vs ~1ms warm). Combined text+image sequence length changes almost every step without
        # this rounding, so cuDNN's shape cache barely ever hits and every attention call in every
        # block pays the cold cost - independent of how many times torch.dynamo itself recompiles.
        if max_seq_length % 16 != 0 and (seq_lengths != seq_lengths[0]).any():
            padded_length = max_seq_length + (16 - max_seq_length % 16)
            text_encoder_output = F.pad(text_encoder_output, (0, 0, 0, padded_length - max_seq_length))
            max_seq_length = padded_length

        bool_attention_mask = torch.arange(max_seq_length, device=tokens_mask.device).unsqueeze(0) < seq_lengths.unsqueeze(1)

        # unflatten (B, T, L*H) → (B, T, L, H) for the transformer - shared by the fresh-encode
        # and cached paths, so callers (Sampler, predict()) never have to do this themselves.
        num_text_layers = len(TEXT_ENCODER_SELECT_LAYERS)
        batch_dim, seq_len, stacked_hidden_dim = text_encoder_output.shape
        text_encoder_output = text_encoder_output.view(batch_dim, seq_len, num_text_layers, stacked_hidden_dim // num_text_layers)

        return (text_encoder_output, bool_attention_mask)

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = latents.shape
        assert frames == 1

        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

        return latents

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, _, channels = latents.shape

        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height * 2, width * 2)

        return latents

    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents / latents_std + latents_mean

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
