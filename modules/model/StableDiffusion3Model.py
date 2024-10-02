from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.model.util.t5_util import encode_t5
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.TrainProgress import TrainProgress

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


class StableDiffusion3ModelEmbedding(BaseModelEmbedding):
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor,
            text_encoder_2_vector: Tensor,
            text_encoder_3_vector: Tensor,
            placeholder: str,
    ):
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder,
        )

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector
        self.text_encoder_3_vector = text_encoder_3_vector


class StableDiffusion3Model(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    tokenizer_3: T5Tokenizer
    noise_scheduler: FlowMatchEulerDiscreteScheduler
    text_encoder_1: CLIPTextModelWithProjection
    text_encoder_2: CLIPTextModelWithProjection
    text_encoder_3: T5EncoderModel
    vae: AutoencoderKL
    transformer: SD3Transformer2DModel

    # autocast context
    autocast_context: torch.autocast | nullcontext
    text_encoder_3_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    text_encoder_3_train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusion3ModelEmbedding | None
    embedding_state: tuple[Tensor, Tensor, Tensor] | None
    additional_embeddings: list[StableDiffusion3ModelEmbedding] | None
    additional_embedding_states: list[tuple[Tensor, Tensor, Tensor] | None]
    embedding_wrapper_1: AdditionalEmbeddingWrapper
    embedding_wrapper_2: AdditionalEmbeddingWrapper
    embedding_wrapper_3: AdditionalEmbeddingWrapper

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    text_encoder_3_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer_1: CLIPTokenizer | None = None,
            tokenizer_2: CLIPTokenizer | None = None,
            tokenizer_3: T5Tokenizer | None = None,
            noise_scheduler: FlowMatchEulerDiscreteScheduler | None = None,
            text_encoder_1: CLIPTextModelWithProjection | None = None,
            text_encoder_2: CLIPTextModelWithProjection | None = None,
            text_encoder_3: T5EncoderModel | None = None,
            vae: AutoencoderKL | None = None,
            transformer: SD3Transformer2DModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: StableDiffusion3ModelEmbedding | None = None,
            embedding_state: tuple[Tensor, Tensor, Tensor] | None = None,
            additional_embeddings: list[StableDiffusion3ModelEmbedding] | None = None,
            additional_embedding_states: list[tuple[Tensor, Tensor, Tensor] | None] = None,
            embedding_wrapper_1: AdditionalEmbeddingWrapper | None = None,
            embedding_wrapper_2: AdditionalEmbeddingWrapper | None = None,
            embedding_wrapper_3: AdditionalEmbeddingWrapper | None = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            text_encoder_3_lora: LoRAModuleWrapper | None = None,
            transformer_lora: LoRAModuleWrapper | None = None,
            lora_state_dict: dict | None = None,
            sd_config: dict | None = None,
            sd_config_filename: str | None = None,
            model_spec: ModelSpec | None = None,
            train_config: TrainConfig | None = None,
    ):
        super().__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
            train_config=train_config,
        )

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.vae = vae
        self.transformer = transformer

        self.autocast_context = nullcontext()
        self.text_encoder_3_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.text_encoder_3_train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper_1 = embedding_wrapper_1
        self.embedding_wrapper_2 = embedding_wrapper_2
        self.embedding_wrapper_3 = embedding_wrapper_3

        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.text_encoder_3_lora = text_encoder_3_lora
        self.transformer_lora = transformer_lora
        self.lora_state_dict = lora_state_dict

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
            self.text_encoder_3.to(device=device)

        if self.text_encoder_3_lora is not None:
            self.text_encoder_3_lora.to(device)

    def transformer_to(self, device: torch.device):
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

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.additional_embeddings, self.embedding, prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int,
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
    ) -> tuple[Tensor, Tensor]:
        # tokenize prompt
        if tokens_1 is None and text is not None and self.tokenizer_1 is not None:
            tokenizer_output = self.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)
            tokens_mask_1 = tokenizer_output.attention_mask.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None and self.tokenizer_2 is not None:
            tokenizer_output = self.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)
            tokens_mask_2 = tokenizer_output.attention_mask.to(self.text_encoder_2.device)

        if tokens_3 is None and text is not None and self.tokenizer_3 is not None:
            tokenizer_output = self.tokenizer_3(
                text,
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
            text_encoder_1_output *= tokens_mask_1[:, :, None]
            text_encoder_2_output *= tokens_mask_2[:, :, None]
            text_encoder_3_output *= tokens_mask_3[:, :, None]

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

        # build the conditioning tensor
        prompt_embedding = torch.concat(
            [text_encoder_1_output, text_encoder_2_output], dim=-1
        )
        prompt_embedding = torch.nn.functional.pad(
            prompt_embedding, (0, text_encoder_3_output.shape[-1] - prompt_embedding.shape[-1])
        )
        prompt_embedding = torch.cat([prompt_embedding, text_encoder_3_output], dim=-2) \
            .to(dtype=self.train_dtype.torch_dtype())
        pooled_prompt_embedding = torch.cat([pooled_text_encoder_1_output, pooled_text_encoder_2_output], dim=-1) \
            .to(dtype=self.train_dtype.torch_dtype())

        return prompt_embedding, pooled_prompt_embedding
