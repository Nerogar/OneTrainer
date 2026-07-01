from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.model.util.llama_util import encode_llama
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert_util import chunk_swap
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLHunyuanVideo,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

DEFAULT_PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
DEFAULT_PROMPT_TEMPLATE_CROP_START = 95

class HunyuanVideoModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor | None,
            text_encoder_2_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_1_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_1_vector,
            is_output_embedding=is_output_embedding,
        )

        self.text_encoder_2_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_2_vector,
            is_output_embedding=False,
        )


class HunyuanVideoModel(BaseModel):
    NATIVE_FPS = 24

    # base model data
    tokenizer_1: LlamaTokenizerFast | None
    tokenizer_2: CLIPTokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder_1: LlamaModel | None
    text_encoder_2: CLIPTextModel | None
    vae: AutoencoderKLHunyuanVideo | None
    transformer: HunyuanVideoTransformer3DModel | None

    # original copies of base model data
    orig_tokenizer_1: LlamaTokenizerFast | None
    orig_tokenizer_2: CLIPTokenizer | None

    # autocast context
    transformer_autocast_context: torch.autocast | nullcontext

    transformer_train_dtype: DataType

    text_encoder_1_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data
    embedding: HunyuanVideoModelEmbedding | None
    additional_embeddings: list[HunyuanVideoModelEmbedding] | None
    embedding_wrapper_1: AdditionalEmbeddingWrapper | None
    embedding_wrapper_2: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.noise_scheduler = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.vae = None
        self.transformer = None

        self.orig_tokenizer_1 = None
        self.orig_tokenizer_2 = None

        self.transformer_autocast_context = nullcontext()

        self.transformer_train_dtype = DataType.FLOAT_32

        self.text_encoder_1_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper_1 = None
        self.embedding_wrapper_2 = None

        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_1_lora,
            self.text_encoder_2_lora,
            self.transformer_lora,
        ] if a is not None]

    def fusion_groups(self) -> list | None:
        # HunyuanVideo fuses qkv in three places: the context-embedder token-refiner blocks, the double
        # blocks (img + txt), and the single blocks (qkv+mlp).
        return [
            ("context_embedder.token_refiner.refiner_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v"], "attn.qkv", "self_attn_qkv"),
            ("transformer_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v"], "attn.qkv", "img_attn_qkv"),
            ("transformer_blocks.{i}", ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], "attn.added_qkv", "txt_attn_qkv"),
            ("single_transformer_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"], "qkv_mlp", "linear1"),
        ]

    def diffusers_to_original(self) -> list | None:
        # The token refiner has its own (non-swapped) adaLN_modulation.1; only the transformer-level
        # final_layer is chunk-swapped.
        return [
            ("context_embedder.time_text_embed.text_embedder.linear_1",   "txt_in.c_embedder.linear_1"),
            ("context_embedder.time_text_embed.text_embedder.linear_2",   "txt_in.c_embedder.linear_2"),
            ("context_embedder.time_text_embed.timestep_embedder.linear_1", "txt_in.t_embedder.mlp.0"),
            ("context_embedder.time_text_embed.timestep_embedder.linear_2", "txt_in.t_embedder.mlp.2"),
            ("context_embedder.proj_in", "txt_in.input_embedder"),
            *chunk_swap("norm_out.linear", "final_layer.adaLN_modulation.1"),
            ("proj_out", "final_layer.linear"),
            ("time_text_embed.guidance_embedder.linear_1", "guidance_in.mlp.0"),
            ("time_text_embed.guidance_embedder.linear_2", "guidance_in.mlp.2"),
            ("time_text_embed.text_embedder.linear_1",     "vector_in.in_layer"),
            ("time_text_embed.text_embedder.linear_2",     "vector_in.out_layer"),
            ("time_text_embed.timestep_embedder.linear_1", "time_in.mlp.0"),
            ("time_text_embed.timestep_embedder.linear_2", "time_in.mlp.2"),
            ("x_embedder.proj", "img_in.proj"),
            ("context_embedder.token_refiner.refiner_blocks.{i}", "txt_in.individual_token_refiner.blocks.{i}", [
                ("attn.qkv",        "self_attn_qkv"),
                ("attn.to_out.0",   "self_attn_proj"),
                ("ff.net.0.proj",   "mlp.fc1"),
                ("ff.net.2",        "mlp.fc2"),
                ("norm_out.linear", "adaLN_modulation.1"),
                ("norm1",           "norm1"),
                ("norm2",           "norm2"),
            ]),
            ("transformer_blocks.{i}", "double_blocks.{i}", [
                ("attn.qkv",                 "img_attn_qkv"),
                ("attn.added_qkv",           "txt_attn_qkv"),
                ("attn.norm_k.weight",       "img_attn_k_norm.weight"),
                ("attn.norm_q.weight",       "img_attn_q_norm.weight"),
                ("attn.to_out.0",            "img_attn_proj"),
                ("ff.net.0.proj",            "img_mlp.fc1"),
                ("ff.net.2",                 "img_mlp.fc2"),
                ("norm1.linear",             "img_mod.linear"),
                ("attn.norm_added_k.weight", "txt_attn_k_norm.weight"),
                ("attn.norm_added_q.weight", "txt_attn_q_norm.weight"),
                ("attn.to_add_out",          "txt_attn_proj"),
                ("ff_context.net.0.proj",    "txt_mlp.fc1"),
                ("ff_context.net.2",         "txt_mlp.fc2"),
                ("norm1_context.linear",     "txt_mod.linear"),
            ]),
            ("single_transformer_blocks.{i}", "single_blocks.{i}", [
                ("qkv_mlp",            "linear1"),
                ("attn.norm_k.weight", "k_norm.weight"),
                ("attn.norm_q.weight", "q_norm.weight"),
                ("proj_out",           "linear2"),
                ("norm.linear",        "modulation.linear"),
            ]),
        ]

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        # HunyuanVideo's two TEs: llama (text_encoder_1) + clip_l (text_encoder_2) (Comfy's HunyuanVideoClipModel).
        return [
            (self.text_encoder_1, {
                ModelFormat.DIFFUSERS_LORA: "text_encoder",
                ModelFormat.KOHYA_LORA: "lora_te1",
                ModelFormat.COMFY_LORA: "text_encoders.llama.transformer",
            }),
            (self.text_encoder_2, {
                ModelFormat.DIFFUSERS_LORA: "text_encoder_2",
                ModelFormat.KOHYA_LORA: "lora_te2",
                ModelFormat.COMFY_LORA: "text_encoders.clip_l.transformer",
            }),
        ]

    def all_embeddings(self) -> list[HunyuanVideoModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_1_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_1_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_1_embedding] if self.embedding is not None else [])

    def all_text_encoder_2_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_2_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_2_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1_to(device=device)
        self.text_encoder_2_to(device=device)

    def text_encoder_1_to(self, device: torch.device):
        if self.text_encoder_1 is not None:
            if self.text_encoder_1_offload_conductor is not None and \
                    self.text_encoder_1_offload_conductor.layer_offload_activated():
                self.text_encoder_1_offload_conductor.to(device)
            else:
                self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device):
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

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
        if self.text_encoder_1 is not None:
            self.text_encoder_1.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
        self.transformer.eval()

    def create_pipeline(self, use_original_tokenizers: bool = False) -> DiffusionPipeline:
        return HunyuanVideoPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            tokenizer=self.orig_tokenizer_1 if use_original_tokenizers else self.tokenizer_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.orig_tokenizer_2 if use_original_tokenizers else self.tokenizer_2,
        )

    def add_text_encoder_1_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_1_embeddings(), prompt)

    def add_text_encoder_2_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_2_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            tokens_mask_1: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            text_encoder_1_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # tokenize prompt
        if tokens_1 is None and text is not None and self.tokenizer_1 is not None:
            llama_text = DEFAULT_PROMPT_TEMPLATE.format(text)

            tokenizer_output = self.tokenizer_1(
                self.add_text_encoder_1_embeddings_to_prompt(llama_text),
                padding='max_length',
                truncation=True,
                max_length=77 + DEFAULT_PROMPT_TEMPLATE_CROP_START,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)
            tokens_mask_1 = tokenizer_output.attention_mask.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None and self.tokenizer_2 is not None:
            tokenizer_output = self.tokenizer_2(
                self.add_text_encoder_2_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)

        text_encoder_1_output, tokens_mask_1, tokens_1 = encode_llama(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-3,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            attention_mask=tokens_mask_1,
            crop_start=DEFAULT_PROMPT_TEMPLATE_CROP_START,
        )
        if text_encoder_1_output is None:
            text_encoder_1_output = torch.zeros(
                size=(batch_size, 77, 768),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )
            tokens_mask_1 = torch.zeros(
                size=(batch_size, 1),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )

        _, pooled_text_encoder_2_output = encode_clip(
            text_encoder=self.text_encoder_2,
            tokens=tokens_2,
            default_layer=-1,
            layer_skip=text_encoder_2_layer_skip,
            add_output=False,
            text_encoder_output=None,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_2_output,
            use_attention_mask=False,
        )
        if pooled_text_encoder_2_output is None:
            pooled_text_encoder_2_output = torch.zeros(
                size=(batch_size, 768),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )

        text_encoder_1_output = self._apply_output_embeddings(
            self.all_text_encoder_1_embeddings(),
            self.tokenizer_1,
            tokens_1,
            text_encoder_1_output,
        )

        # apply dropout
        if text_encoder_1_dropout_probability is not None and text_encoder_1_dropout_probability > 0.0:
            dropout_text_encoder_1_mask = (torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask[:, None, None]

        if text_encoder_2_dropout_probability is not None and text_encoder_2_dropout_probability > 0.0:
            dropout_text_encoder_2_mask = (torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask[:, None]

        return text_encoder_1_output, pooled_text_encoder_2_output, tokens_mask_1
