from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
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
    DDIMScheduler,
    DiffusionPipeline,
    PixArtAlphaPipeline,
    PixArtSigmaPipeline,
    Transformer2DModel,
)
from transformers import T5EncoderModel, T5Tokenizer


class PixArtAlphaModelEmbedding(BaseModelEmbedding):
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor,
            placeholder: str,
    ):
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_vector.shape[0],
            placeholder=placeholder,
        )

        self.text_encoder_vector = text_encoder_vector


class PixArtAlphaModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer: T5Tokenizer
    noise_scheduler: DDIMScheduler
    text_encoder: T5EncoderModel
    vae: AutoencoderKL
    transformer: Transformer2DModel

    # autocast context
    autocast_context: torch.autocast | nullcontext
    text_encoder_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    text_encoder_train_dtype: DataType

    # persistent embedding training data
    embedding: PixArtAlphaModelEmbedding | None
    embedding_state: Tensor | None
    additional_embeddings: list[PixArtAlphaModelEmbedding] | None
    additional_embedding_states: list[Tensor | None]
    embedding_wrapper: AdditionalEmbeddingWrapper

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer: T5Tokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder: T5EncoderModel | None = None,
            vae: AutoencoderKL | None = None,
            transformer: Transformer2DModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: PixArtAlphaModelEmbedding | None = None,
            embedding_state: Tensor | None = None,
            additional_embeddings: list[PixArtAlphaModelEmbedding] | None = None,
            additional_embedding_states: list[Tensor | None] = None,
            embedding_wrapper: AdditionalEmbeddingWrapper | None = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            transformer_lora: LoRAModuleWrapper | None = None,
            lora_state_dict: dict | None = None,
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

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer

        self.autocast_context = nullcontext()
        self.text_encoder_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper = embedding_wrapper

        self.text_encoder_lora = text_encoder_lora
        self.transformer_lora = transformer_lora
        self.lora_state_dict = lora_state_dict

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

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
        self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        match self.model_type:
            case ModelType.PIXART_ALPHA:
                return PixArtAlphaPipeline(
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=self.noise_scheduler,
                )
            case ModelType.PIXART_SIGMA:
                return PixArtSigmaPipeline(
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=self.noise_scheduler,
                )

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.additional_embeddings, self.embedding, prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int,
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
                text,
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

        # apply dropout
        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            attention_mask = attention_mask * dropout_text_encoder_mask[:, None]
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output, attention_mask
