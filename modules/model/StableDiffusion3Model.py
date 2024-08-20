from contextlib import nullcontext
from uuid import uuid4

from modules.model.BaseModel import BaseModel
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


class StableDiffusion3ModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor,
            text_encoder_2_vector: Tensor,
            text_encoder_3_vector: Tensor,
            placeholder: str,
    ):
        token_count = text_encoder_1_vector.shape[0]

        self.uuid = uuid
        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector
        self.text_encoder_3_vector = text_encoder_3_vector
        self.placeholder = placeholder
        self.text_tokens = [f"<{uuid4()}>" for _ in range(token_count)]


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

    train_dtype: DataType

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

    # def force_v_prediction(self):
    #     self.noise_scheduler.config.prediction_type = 'v_prediction'
    #     self.sd_config['model']['params']['parameterization'] = 'v'
    #
    # def force_epsilon_prediction(self):
    #     self.noise_scheduler.config.prediction_type = 'epsilon'
    #     self.sd_config['model']['params']['parameterization'] = 'epsilon'
    #
    # def rescale_noise_scheduler_to_zero_terminal_snr(self):
    #     rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        for embedding in self.additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        if self.embedding is not None:
            embedding_string = ''.join(self.embedding.text_tokens)
            prompt = prompt.replace(self.embedding.placeholder, embedding_string)

        return prompt
