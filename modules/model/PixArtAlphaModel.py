from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.t5_util import encode_t5
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PixArtAlphaPipeline,
    PixArtSigmaPipeline,
    Transformer2DModel,
)
from transformers import T5EncoderModel, T5Tokenizer


class PixArtAlphaModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )


class PixArtAlphaModel(BaseModel):
    # base model data
    tokenizer: T5Tokenizer | None
    orig_tokenizer: T5Tokenizer | None
    noise_scheduler: DDIMScheduler | None
    text_encoder: T5EncoderModel | None
    vae: AutoencoderKL | None
    transformer: Transformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data
    embedding: PixArtAlphaModelEmbedding | None
    embedding_state: Tensor | None
    additional_embeddings: list[PixArtAlphaModelEmbedding] | None
    embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.orig_tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def fusion_groups(self) -> list | None:
        # PixArt fuses TWO attentions differently: self-attention fuses q/k/v (3 leaves), while cross-attention
        # fuses ONLY k/v (2 leaves) into kv_linear; fuse_split is arity-agnostic so both groups just work.
        return [
            ("transformer_blocks.{i}", ["attn1.to_q", "attn1.to_k", "attn1.to_v"], "attn1.qkv", "attn.qkv"),
            ("transformer_blocks.{i}", ["attn2.to_k", "attn2.to_v"], "attn2.kv", "cross_attn.kv_linear"),
        ]

    def diffusers_to_original(self) -> list | None:
        # PixArt has NO adaLN chunk swap: its adaLN-single is a scale_shift_table (nn.Parameter, not Linear);
        # those rules are present for the full-model path and inert for LoRA. pos_embed is GENERATED (not
        # key-mapped), so convert_pixart_diffusers_to_ckpt adds it after this body runs.
        return [
            ("adaln_single.emb.aspect_ratio_embedder.linear_1", "ar_embedder.mlp.0"),
            ("adaln_single.emb.aspect_ratio_embedder.linear_2", "ar_embedder.mlp.2"),
            ("adaln_single.emb.resolution_embedder.linear_1",   "csize_embedder.mlp.0"),
            ("adaln_single.emb.resolution_embedder.linear_2",   "csize_embedder.mlp.2"),
            ("caption_projection.linear_1", "y_embedder.y_proj.fc1"),
            ("caption_projection.linear_2", "y_embedder.y_proj.fc2"),
            ("pos_embed.proj", "x_embedder.proj"),
            ("adaln_single.emb.timestep_embedder.linear_1", "t_embedder.mlp.0"),
            ("adaln_single.emb.timestep_embedder.linear_2", "t_embedder.mlp.2"),
            ("adaln_single.linear", "t_block.1"),
            ("proj_out", "final_layer.linear"),
            ("scale_shift_table", "final_layer.scale_shift_table"),
            ("transformer_blocks.{i}", "blocks.{i}", [
                ("attn1.qkv",       "attn.qkv"),
                ("attn2.kv",        "cross_attn.kv_linear"),
                ("attn1.to_out.0",  "attn.proj"),
                ("attn2.to_q",      "cross_attn.q_linear"),
                ("attn2.to_out.0",  "cross_attn.proj"),
                ("ff.net.0.proj",   "mlp.fc1"),
                ("ff.net.2",        "mlp.fc2"),
                ("scale_shift_table", "scale_shift_table"),
            ]),
        ]

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        # Single T5 text encoder (Comfy's PixArt TE is a single t5xxl).
        return [
            (self.text_encoder, {
                ModelFormat.DIFFUSERS_LORA: "text_encoder",
                ModelFormat.KOHYA_LORA: "lora_te1",
                ModelFormat.COMFY_LORA: "text_encoders.t5xxl.transformer",
            }),
        ]

    def all_embeddings(self) -> list[PixArtAlphaModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
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
        self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self, use_original_tokenizers: bool = False) -> DiffusionPipeline:
        tokenizer = self.orig_tokenizer if use_original_tokenizers else self.tokenizer
        match self.model_type:
            case ModelType.PIXART_ALPHA:
                return PixArtAlphaPipeline(
                    tokenizer=tokenizer,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=self.noise_scheduler,
                )
            case ModelType.PIXART_SIGMA:
                return PixArtSigmaPipeline(
                    tokenizer=tokenizer,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=self.noise_scheduler,
                )

    def add_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
            attention_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            max_token_length = 120
            # deactivated for performance reasons. most people don't need 300 tokens
            # if self.model_type.is_pixart_sigma():
            #     max_token_length = 300

            tokenizer_output = self.tokenizer(
                self.add_text_encoder_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=max_token_length,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)

            attention_mask = tokenizer_output.attention_mask
            attention_mask = attention_mask.to(self.text_encoder.device)

        with self.text_encoder_autocast_context:
            text_encoder_output = encode_t5(
                text_encoder=self.text_encoder,
                tokens=tokens,
                default_layer=-1,
                layer_skip=text_encoder_layer_skip,
                text_encoder_output=text_encoder_output,
                attention_mask=attention_mask,
            )

        text_encoder_output = self._apply_output_embeddings(
            self.all_text_encoder_embeddings(),
            self.tokenizer,
            tokens,
            text_encoder_output,
        )

        # apply dropout
        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            attention_mask = attention_mask * dropout_text_encoder_mask[:, None]
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output, attention_mask
