from contextlib import nullcontext

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class StableDiffusionXLModelEmbedding(BaseModelEmbedding):
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor,
            text_encoder_2_vector: Tensor,
            placeholder: str,
    ):
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder,
        )

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector


class StableDiffusionXLModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    vae: AutoencoderKL
    unet: UNet2DConditionModel

    # autocast context
    autocast_context: torch.autocast | nullcontext
    vae_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    vae_train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusionXLModelEmbedding | None
    embedding_state: tuple[Tensor, Tensor] | None
    additional_embeddings: list[StableDiffusionXLModelEmbedding] | None
    additional_embedding_states: list[tuple[Tensor, Tensor] | None]
    embedding_wrapper_1: AdditionalEmbeddingWrapper
    embedding_wrapper_2: AdditionalEmbeddingWrapper

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer_1: CLIPTokenizer | None = None,
            tokenizer_2: CLIPTokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder_1: CLIPTextModel | None = None,
            text_encoder_2: CLIPTextModelWithProjection | None = None,
            vae: AutoencoderKL | None = None,
            unet: UNet2DConditionModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: StableDiffusionXLModelEmbedding | None = None,
            embedding_state: tuple[Tensor, Tensor] | None = None,
            additional_embeddings: list[StableDiffusionXLModelEmbedding] | None = None,
            additional_embedding_states: list[tuple[Tensor, Tensor] | None] = None,
            embedding_wrapper_1: AdditionalEmbeddingWrapper | None = None,
            embedding_wrapper_2: AdditionalEmbeddingWrapper | None = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
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
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.unet = unet

        self.autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.vae_train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper_1 = embedding_wrapper_1
        self.embedding_wrapper_1 = embedding_wrapper_2

        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.unet_lora = unet_lora
        self.lora_state_dict = lora_state_dict

        self.sd_config = sd_config
        self.sd_config_filename = sd_config_filename

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)
        self.text_encoder_2.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def text_encoder_1_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device):
        self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.additional_embeddings, self.embedding, prompt)

    def encode_text(
            self,
            text: str = None,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
    ):
        if tokens_1 is None and text is not None:
            tokenizer_output = self.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokenizer_output = self.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)

        text_encoder_1_output, _ = encode_clip(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-2,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            add_pooled_output=False,
            use_attention_mask=False,
            add_layer_norm=False,
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

        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output
