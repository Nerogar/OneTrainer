import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AnimaAutoBlocks,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

PROMPT_MAX_LENGTH = 512


# Maps the diffusers CosmosTransformer3DModel state dict back to the original Anima checkpoint keys.
# This is the exact inverse of the forward conversion in diffusers' scripts/convert_anima_to_diffusers.py
# (which delegates the transformer to convert_cosmos_to_diffusers.convert_transformer with
# TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0).
# The original keys carry a "net." prefix; the conversion is a flat 1:1 rename, no tensor fusion.
def diffusers_to_original():
    return [
        ("patch_embed.proj",   "net.x_embedder.proj.1"),
        ("time_embed.t_embedder", "net.t_embedder.1"),
        ("time_embed.norm",    "net.t_embedding_norm"),
        ("norm_out.linear_1",  "net.final_layer.adaln_modulation.1"),
        ("norm_out.linear_2",  "net.final_layer.adaln_modulation.2"),
        ("proj_out",           "net.final_layer.linear"),
        ("transformer_blocks.{i}", "net.blocks.{i}", [
            ("norm1.linear_1", "adaln_modulation_self_attn.1"),
            ("norm1.linear_2", "adaln_modulation_self_attn.2"),
            ("attn1.norm_q",   "self_attn.q_norm"),
            ("attn1.norm_k",   "self_attn.k_norm"),
            ("attn1.to_q",     "self_attn.q_proj"),
            ("attn1.to_k",     "self_attn.k_proj"),
            ("attn1.to_v",     "self_attn.v_proj"),
            ("attn1.to_out.0", "self_attn.output_proj"),
            ("norm2.linear_1", "adaln_modulation_cross_attn.1"),
            ("norm2.linear_2", "adaln_modulation_cross_attn.2"),
            ("attn2.norm_q",   "cross_attn.q_norm"),
            ("attn2.norm_k",   "cross_attn.k_norm"),
            ("attn2.to_q",     "cross_attn.q_proj"),
            ("attn2.to_k",     "cross_attn.k_proj"),
            ("attn2.to_v",     "cross_attn.v_proj"),
            ("attn2.to_out.0", "cross_attn.output_proj"),
            ("norm3.linear_1", "adaln_modulation_mlp.1"),
            ("norm3.linear_2", "adaln_modulation_mlp.2"),
            ("ff.net.0.proj",  "mlp.layer1"),
            ("ff.net.2",       "mlp.layer2"),
        ]),
    ]

diffusers_checkpoint_to_original = diffusers_to_original()


class AnimaModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    t5_tokenizer: T5TokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3Model | None
    text_conditioner: AnimaTextConditioner | None
    vae: AutoencoderKLQwenImage | None
    transformer: CosmosTransformer3DModel | None

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
        self.t5_tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.text_conditioner = None
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

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device): #TODO share more code between models
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)
            self.text_conditioner.to(device=device)

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

    def release(self):
        temp_device = torch.device(self.train_config.temp_device)
        self.vae_to(temp_device)
        self.text_encoder_to(temp_device)
        self.transformer_to(temp_device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
            self.text_conditioner.eval()
        self.transformer.eval()

    def create_pipeline(self):
        pipe = AnimaAutoBlocks().init_pipeline()
        pipe.update_components(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            text_conditioner=self.text_conditioner,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.noise_scheduler,
        )
        return pipe

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
    ) -> Tensor:
        # Two-stage encoding: Qwen3 text encoder → AnimaTextConditioner (with T5 token ids as queries).
        # text_encoder_output, when provided from cache, is already the conditioner output.
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

            t5_output = self.t5_tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            t5_ids = t5_output.input_ids.to(self.text_encoder.device)
            t5_mask = t5_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                qwen_hidden = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    output_hidden_states=False,
                ).last_hidden_state
                # zero out padding positions (mirrors diffusers AnimaTextEncoderStep)
                qwen_hidden = qwen_hidden * tokens_mask.to(qwen_hidden).unsqueeze(-1)
                text_encoder_output = self.text_conditioner(
                    source_hidden_states=qwen_hidden.to(dtype=self.text_conditioner.dtype),
                    target_input_ids=t5_ids,
                    target_attention_mask=t5_mask,
                    source_attention_mask=tokens_mask,
                )

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # https://github.com/Nerogar/OneTrainer/issues/957

        # conditioner output is always (B, 512, 1024) and fully dense (zeros for padding positions);
        # the Cosmos transformer takes encoder_hidden_states with no separate text attention mask.
        return text_encoder_output

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
