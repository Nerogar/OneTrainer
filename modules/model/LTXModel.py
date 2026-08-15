import math
from contextlib import nullcontext

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert_util import add_prefix
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    GemmaTokenizer,
    GemmaTokenizerFast,
)

PROMPT_MAX_LENGTH = 1024
# Gemma expects left padding for chat-style prompts, matching LTX2Pipeline._get_gemma_prompt_embeds. The
# checkpoint's tokenizer_config.json doesn't set it (transformers defaults to "right").
PROMPT_PADDING_SIDE = "left"


class LTXModel(BaseModel):
    NATIVE_FPS = 24

    # base model data
    # LTX 2.3 names the fast tokenizer and the Gemma 3 encoder, LTX 2.5 the slow tokenizer and the Gemma 4 one
    tokenizer: GemmaTokenizer | GemmaTokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration | None
    vae: AutoencoderKLLTX2Video | None
    connectors: LTX2TextConnectors | None
    transformer: LTX2VideoTransformer3DModel | None
    low_noise_transformer: LTX2VideoTransformer3DModel | None

    # audio branch - frozen, never run
    audio_vae: AutoencoderKLLTX2Audio | None
    vocoder: LTX2Vocoder | LTX2VocoderWithBWE | None

    transformer_autocast_context: torch.autocast | nullcontext
    low_noise_transformer_autocast_context: torch.autocast | nullcontext
    text_encoder_autocast_context: torch.autocast | nullcontext
    connectors_autocast_context: torch.autocast | nullcontext
    transformer_train_dtype: DataType
    low_noise_transformer_train_dtype: DataType
    text_encoder_train_dtype: DataType
    connectors_train_dtype: DataType

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
        self.connectors = None
        self.transformer = None
        self.low_noise_transformer = None

        self.audio_vae = None
        self.vocoder = None

        self.transformer_autocast_context = nullcontext()
        self.low_noise_transformer_autocast_context = nullcontext()
        self.text_encoder_autocast_context = nullcontext()
        self.connectors_autocast_context = nullcontext()
        self.transformer_train_dtype = DataType.FLOAT_32
        self.low_noise_transformer_train_dtype = DataType.FLOAT_32
        self.text_encoder_train_dtype = DataType.FLOAT_32
        self.connectors_train_dtype = DataType.FLOAT_32

        self.transformer_lora = None
        self.lora_state_dict = None

    @staticmethod
    def _attn(name: str) -> tuple:
        # one attention module's leaves. Only the qk norms are renamed; the projections are spelled the same in
        # both namespaces and are listed so the block rule stays strict -- a leaf with no rule is an error, not a
        # silent passthrough.
        return (name, name, [
            ("norm_q",         "q_norm"),
            ("norm_k",         "k_norm"),
            ("to_q",           "to_q"),
            ("to_k",           "to_k"),
            ("to_v",           "to_v"),
            ("to_out.0",       "to_out.0"),
            ("to_gate_logits", "to_gate_logits"),
        ])

    def diffusers_to_original(self) -> list | None:
        # rename only -- LTX-2's native (Lightricks/ComfyUI) layout differs from diffusers in module names alone.
        # The two namespaces name the same modules, so this one body serves both the full checkpoint and a LoRA;
        # they differ only in the top prefix each format adds (see checkpoint_diffusers_to_original).
        return [
            ("proj_in",                          "patchify_proj"),
            ("audio_proj_in",                    "audio_patchify_proj"),
            ("proj_out",                         "proj_out"),
            ("audio_proj_out",                   "audio_proj_out"),
            # every modulation predictor is an "adaln_single" natively; diffusers names each after what it feeds
            ("time_embed",                       "adaln_single"),
            ("audio_time_embed",                 "audio_adaln_single"),
            ("prompt_adaln",                     "prompt_adaln_single"),
            ("audio_prompt_adaln",               "audio_prompt_adaln_single"),
            ("av_cross_attn_video_scale_shift",  "av_ca_video_scale_shift_adaln_single"),
            ("av_cross_attn_video_a2v_gate",     "av_ca_a2v_gate_adaln_single"),
            ("av_cross_attn_audio_scale_shift",  "av_ca_audio_scale_shift_adaln_single"),
            ("av_cross_attn_audio_v2a_gate",     "av_ca_v2a_gate_adaln_single"),
            ("scale_shift_table",                "scale_shift_table"),
            ("audio_scale_shift_table",          "audio_scale_shift_table"),
            # LTX 2.5 only
            ("keyframes_abs_pos_embedding",      "keyframes_abs_pos_embedding"),
            ("transformer_blocks.{i}", "transformer_blocks.{i}", [
                ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
                ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
                ("scale_shift_table",                      "scale_shift_table"),
                ("audio_scale_shift_table",                "audio_scale_shift_table"),
                ("prompt_scale_shift_table",               "prompt_scale_shift_table"),
                ("audio_prompt_scale_shift_table",         "audio_prompt_scale_shift_table"),
                self._attn("attn1"),
                self._attn("attn2"),
                self._attn("audio_attn1"),
                self._attn("audio_attn2"),
                self._attn("audio_to_video_attn"),
                self._attn("video_to_audio_attn"),
                ("ff",       "ff"),
                ("audio_ff", "audio_ff"),
            ]),
        ]

    def checkpoint_diffusers_to_original(self) -> list | None:
        # the full checkpoint nests the transformer under model.diffusion_model., beside the vae/text-projection
        # branches of the same file (a LoRA carries only diffusion_model., which the saver adds).
        return [self.diffusers_to_original(), add_prefix("model.diffusion_model")]

    def create_pipeline(self) -> DiffusionPipeline:
        return LTX2Pipeline(
            scheduler=self.noise_scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=self.vocoder,
        )

    def encode_text(
            self,
            train_device: torch.device,
            text: str | list[str] | None = None,
            connector_video_embeds: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
    ) -> Tensor:
        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # needs a cached null-caption embedding, not zero-out

        if connector_video_embeds is not None:
            return connector_video_embeds

        text_encoder_outputs, tokens_mask = self.encode_text_encoder(text=text)
        return self.encode_connectors(text_encoder_outputs, tokens_mask, train_device)[0]

    def encode_text_encoder(
            self,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokenizer_output = self.tokenizer(
                [t.strip() for t in text],
                padding='max_length',
                padding_side=PROMPT_PADDING_SIDE,
                max_length=PROMPT_MAX_LENGTH,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        # every layer's hidden state, incl. the embedding layer, flattened to 3D - the "Pack to 3D" step in
        # LTX2Pipeline._get_gemma_prompt_embeds
        text_encoder_outputs = []
        with self.text_encoder_autocast_context:
            for i in range(tokens.shape[0]):
                output = self.text_encoder(
                    tokens[i:i + 1],
                    attention_mask=tokens_mask[i:i + 1],
                    output_hidden_states=True,
                    use_cache=False,
                )
                stacked = torch.stack(output.hidden_states, dim=-1).flatten(2, 3)  # [1, T, H*L]
                # Gemma's residual stream comes out fp32 under autocast; downcast to the TE dtype, which
                # halves every full-width copy downstream
                text_encoder_outputs.append(stacked.to(self.text_encoder_train_dtype.torch_dtype()))
                del output, stacked  # drop the retained per-layer hidden_states tuple before the next prompt

        return text_encoder_outputs, tokens_mask

    def encode_connectors(
            self,
            text_encoder_outputs: list[Tensor],
            tokens_mask: Tensor,
            train_device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        video_embeds, audio_embeds = [], []
        for i, text_encoder_output in enumerate(text_encoder_outputs):
            video_embed, audio_embed, mask = self.connectors(
                text_encoder_output.to(train_device), tokens_mask[i:i + 1].to(train_device),
                padding_side=PROMPT_PADDING_SIDE,
            )
            # the connectors replace padded positions with learnable registers, so the mask they return is
            # all-attend. _assert_async queues the check as a kernel, so it costs no device sync.
            torch._assert_async(mask.all(), "connector attention mask is not all-True")
            video_embeds.append(video_embed)
            audio_embeds.append(audio_embed)
        return torch.cat(video_embeds), torch.cat(audio_embeds)

    # video latents [B, C, F, H, W] <-> patch sequence [B, S, D]
    @staticmethod
    def pack_latents(latents: Tensor, patch_size: int = 1, patch_size_t: int = 1) -> Tensor:
        return LTX2Pipeline._pack_latents(latents, patch_size, patch_size_t)

    @staticmethod
    def unpack_latents(
            latents: Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1,
    ) -> Tensor:
        return LTX2Pipeline._unpack_latents(latents, num_frames, height, width, patch_size, patch_size_t)

    # audio latents [B, C, L, M] -> patch sequence [B, S, D]. Without a patch size the pipeline packs all mel
    # bins into one patch, which is what the checkpoint's audio_patch_size config asks for.
    @staticmethod
    def pack_audio_latents(latents: Tensor) -> Tensor:
        return LTX2Pipeline._pack_audio_latents(latents)

    def scale_latents(self, latents: Tensor) -> Tensor:
        return LTX2Pipeline._normalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor)

    def unscale_latents(self, latents: Tensor) -> Tensor:
        return LTX2Pipeline._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor)

    def calculate_timestep_shift(self, num_latent_frames: int, latent_height: int, latent_width: int) -> float:
        # resolution/length-dependent flow-matching shift, matching Lightricks' get_normal_shift. patch_size
        # is 1, so the latent element count is the transformer sequence length.
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift

        image_seq_len = num_latent_frames * latent_height * latent_width
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
