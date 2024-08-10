from contextlib import nullcontext
from uuid import uuid4

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DDIMScheduler, Transformer2DModel, \
    PixArtAlphaPipeline, PixArtSigmaPipeline
from torch import Tensor
from transformers import T5Tokenizer, \
    T5EncoderModel

from modules.model.BaseModel import BaseModel
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class PixArtAlphaModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor,
            placeholder: str,
    ):
        token_count = text_encoder_vector.shape[0]

        self.uuid = uuid
        self.text_encoder_vector = text_encoder_vector
        self.placeholder = placeholder
        self.text_tokens = [f"<{uuid4()}>" for _ in range(token_count)]


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
        super(PixArtAlphaModel, self).__init__(
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
        for embedding in self.additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        if self.embedding is not None:
            embedding_string = ''.join(self.embedding.text_tokens)
            prompt = prompt.replace(self.embedding.placeholder, embedding_string)

        return prompt
