from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.model.util.t5_util import encode_t5
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class StableDiffusion3ModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor | None,
            text_encoder_2_vector: Tensor | None,
            text_encoder_3_vector: Tensor | None,
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
            is_output_embedding=is_output_embedding,
        )

        self.text_encoder_3_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_3_vector,
            is_output_embedding=is_output_embedding,
        )


class StableDiffusion3Model(BaseModel):
    # base model data
    tokenizer_1: CLIPTokenizer | None
    tokenizer_2: CLIPTokenizer | None
    tokenizer_3: T5Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder_1: CLIPTextModelWithProjection | None
    text_encoder_2: CLIPTextModelWithProjection | None
    text_encoder_3: T5EncoderModel | None
    vae: AutoencoderKL | None
    transformer: SD3Transformer2DModel | None

    # autocast context
    text_encoder_3_autocast_context: torch.autocast | nullcontext

    text_encoder_3_train_dtype: DataType

    text_encoder_3_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data
    embedding: StableDiffusion3ModelEmbedding | None
    additional_embeddings: list[StableDiffusion3ModelEmbedding] | None
    embedding_wrapper_1: AdditionalEmbeddingWrapper | None
    embedding_wrapper_2: AdditionalEmbeddingWrapper | None
    embedding_wrapper_3: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    text_encoder_3_lora: LoRAModuleWrapper | None
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
        self.tokenizer_3 = None
        self.noise_scheduler = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.vae = None
        self.transformer = None

        self.text_encoder_3_autocast_context = nullcontext()

        self.text_encoder_3_train_dtype = DataType.FLOAT_32

        self.text_encoder_3_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper_1 = None
        self.embedding_wrapper_2 = None
        self.embedding_wrapper_3 = None

        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.text_encoder_3_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_1_lora,
            self.text_encoder_2_lora,
            self.text_encoder_3_lora,
            self.transformer_lora,
        ] if a is not None]

    def all_embeddings(self) -> list[StableDiffusion3ModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_1_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_1_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_1_embedding] if self.embedding is not None else [])

    def all_text_encoder_2_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_2_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_2_embedding] if self.embedding is not None else [])

    def all_text_encoder_3_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_3_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_3_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1_to(device=device)
        self.text_encoder_2_to(device=device)
        self.text_encoder_3_to(device=device)

    def text_encoder_1_to(self, device: torch.device):
        if self.text_encoder_1 is not None:
            self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device):
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def text_encoder_3_to(self, device: torch.device):
        if self.text_encoder_3 is not None:
            if self.text_encoder_3_offload_conductor is not None and \
                    self.text_encoder_3_offload_conductor.layer_offload_activated():
                self.text_encoder_3_offload_conductor.to(device)
            else:
                self.text_encoder_3.to(device=device)

        if self.text_encoder_3_lora is not None:
            self.text_encoder_3_lora.to(device)

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
        if self.text_encoder_3 is not None:
            self.text_encoder_3.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            tokenizer=self.tokenizer_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            text_encoder_3=self.text_encoder_3,
            tokenizer_3=self.tokenizer_3,
        )

    def add_text_encoder_1_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_1_embeddings(), prompt)

    def add_text_encoder_2_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_2_embeddings(), prompt)

    def add_text_encoder_3_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_3_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            tokens_3: Tensor = None,
            tokens_mask_1: Tensor = None,
            tokens_mask_2: Tensor = None,
            tokens_mask_3: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_3_layer_skip: int = 0,
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            text_encoder_3_dropout_probability: float | None = None,
            apply_attention_mask: bool = False,
            text_encoder_1_output: Tensor = None,
            pooled_text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
            text_encoder_3_output: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # tokenize prompt
        if tokens_1 is None and text is not None and self.tokenizer_1 is not None:
            tokenizer_output = self.tokenizer_1(
                self.add_text_encoder_1_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=77,
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
            tokens_mask_2 = tokenizer_output.attention_mask.to(self.text_encoder_2.device)

        if tokens_3 is None and text is not None and self.tokenizer_3 is not None:
            tokenizer_output = self.tokenizer_3(
                self.add_text_encoder_3_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_3 = tokenizer_output.input_ids.to(self.text_encoder_3.device)
            tokens_mask_3 = tokenizer_output.attention_mask.to(self.text_encoder_3.device)

        text_encoder_1_output, pooled_text_encoder_1_output = encode_clip(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-2,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_1_output,
            use_attention_mask=False,
            add_layer_norm=False,
        )
        if text_encoder_1_output is None or pooled_text_encoder_1_output is None:
            pooled_text_encoder_1_output = torch.zeros(
                size=(batch_size, 768),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )
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

        text_encoder_2_output, pooled_text_encoder_2_output = encode_clip(
            text_encoder=self.text_encoder_2,
            tokens=tokens_2,
            default_layer=-2,
            layer_skip=text_encoder_2_layer_skip,
            text_encoder_output=text_encoder_2_output,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_2_output,
            use_attention_mask=False,
            add_layer_norm=False,
        )
        if text_encoder_2_output is None or pooled_text_encoder_2_output is None:
            pooled_text_encoder_2_output = torch.zeros(
                size=(batch_size, 1280),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )
            text_encoder_2_output = torch.zeros(
                size=(batch_size, 77, 1280),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )
            tokens_mask_2 = torch.zeros(
                size=(batch_size, 1),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )

        with self.text_encoder_3_autocast_context:
            text_encoder_3_output = encode_t5(
                text_encoder=self.text_encoder_3,
                tokens=tokens_3,
                default_layer=-1,
                layer_skip=text_encoder_3_layer_skip,
                text_encoder_output=text_encoder_3_output,
                use_attention_mask=False,
                attention_mask=tokens_mask_3,
            )
            if text_encoder_3_output is None:
                text_encoder_3_output = torch.zeros(
                    size=(batch_size, 77, self.transformer.config.joint_attention_dim),
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                )
                tokens_mask_3 = torch.zeros(
                    size=(batch_size, 1),
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                )

        if apply_attention_mask:
            text_encoder_1_output = text_encoder_1_output * tokens_mask_1[:, :, None]
            text_encoder_2_output = text_encoder_2_output * tokens_mask_2[:, :, None]
            text_encoder_3_output = text_encoder_3_output * tokens_mask_3[:, :, None]

        text_encoder_1_output = self._apply_output_embeddings(
            self.all_text_encoder_1_embeddings(),
            self.tokenizer_1,
            tokens_1,
            text_encoder_1_output,
        )

        text_encoder_2_output = self._apply_output_embeddings(
            self.all_text_encoder_2_embeddings(),
            self.tokenizer_2,
            tokens_2,
            text_encoder_2_output,
        )

        text_encoder_3_output = self._apply_output_embeddings(
            self.all_text_encoder_3_embeddings(),
            self.tokenizer_3,
            tokens_3,
            text_encoder_3_output,
        )

        # apply dropout
        if text_encoder_1_dropout_probability is not None:
            dropout_text_encoder_1_mask = (torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask[:, None, None]
            pooled_text_encoder_1_output = pooled_text_encoder_1_output * dropout_text_encoder_1_mask[:, None]

        if text_encoder_2_dropout_probability is not None:
            dropout_text_encoder_2_mask = (torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask[:, None, None]
            pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask[:, None]

        if text_encoder_3_dropout_probability is not None:
            dropout_text_encoder_3_mask = (torch.tensor(
                [rand.random() > text_encoder_3_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_3_output = text_encoder_3_output * dropout_text_encoder_3_mask[:, None, None]

        return text_encoder_1_output, text_encoder_2_output, text_encoder_3_output, pooled_text_encoder_1_output, pooled_text_encoder_2_output

    def combine_text_encoder_output(
            self,
            text_encoder_1_output: Tensor,
            text_encoder_2_output: Tensor,
            text_encoder_3_output: Tensor,
            pooled_text_encoder_1_output: Tensor,
            pooled_text_encoder_2_output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prompt_embedding = torch.concat(
            [text_encoder_1_output, text_encoder_2_output], dim=-1
        )
        prompt_embedding = torch.nn.functional.pad(
            prompt_embedding, (0, text_encoder_3_output.shape[-1] - prompt_embedding.shape[-1])
        )
        prompt_embedding = torch.cat([prompt_embedding, text_encoder_3_output], dim=-2)
        pooled_prompt_embedding = torch.cat([pooled_text_encoder_1_output, pooled_text_encoder_2_output], dim=-1)

        return prompt_embedding, pooled_prompt_embedding
