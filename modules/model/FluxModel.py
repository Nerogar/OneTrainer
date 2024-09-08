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
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class FluxModelEmbedding(BaseModelEmbedding):
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


class FluxModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: T5Tokenizer
    noise_scheduler: FlowMatchEulerDiscreteScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: T5EncoderModel
    vae: AutoencoderKL
    transformer: FluxTransformer2DModel

    # autocast context
    autocast_context: torch.autocast | nullcontext
    text_encoder_2_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    text_encoder_2_train_dtype: DataType

    # persistent embedding training data
    embedding: FluxModelEmbedding | None
    embedding_state: tuple[Tensor, Tensor, Tensor] | None
    additional_embeddings: list[FluxModelEmbedding] | None
    additional_embedding_states: list[tuple[Tensor, Tensor, Tensor] | None]
    embedding_wrapper_1: AdditionalEmbeddingWrapper
    embedding_wrapper_2: AdditionalEmbeddingWrapper

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer_1: CLIPTokenizer | None = None,
            tokenizer_2: T5Tokenizer | None = None,
            noise_scheduler: FlowMatchEulerDiscreteScheduler | None = None,
            text_encoder_1: CLIPTextModel | None = None,
            text_encoder_2: T5EncoderModel | None = None,
            vae: AutoencoderKL | None = None,
            transformer: FluxTransformer2DModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: FluxModelEmbedding | None = None,
            embedding_state: tuple[Tensor, Tensor, Tensor] | None = None,
            additional_embeddings: list[FluxModelEmbedding] | None = None,
            additional_embedding_states: list[tuple[Tensor, Tensor, Tensor] | None] = None,
            embedding_wrapper_1: AdditionalEmbeddingWrapper | None = None,
            embedding_wrapper_2: AdditionalEmbeddingWrapper | None = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            transformer_lora: LoRAModuleWrapper | None = None,
            lora_state_dict: dict | None = None,
            model_spec: ModelSpec | None = None,
            train_config: TrainConfig | None = None,
    ):
        super(FluxModel, self).__init__(
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
        self.transformer = transformer

        self.autocast_context = nullcontext()
        self.text_encoder_2_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.text_encoder_2_train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper_1 = embedding_wrapper_1
        self.embedding_wrapper_2 = embedding_wrapper_2

        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.transformer_lora = transformer_lora
        self.lora_state_dict = lora_state_dict

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1_to(device=device)
        self.text_encoder_2_to(device=device)

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
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return FluxPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            tokenizer=self.tokenizer_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
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
            tokens_mask_2: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            apply_attention_mask: bool = False,
            pooled_text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # tokenize prompt
        if tokens_1 is None and text is not None and self.tokenizer_1 is not None:
            tokenizer_output = self.tokenizer_1(
                text,
                # padding='max_length',
                truncation=False,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None and self.tokenizer_2 is not None:
            tokenizer_output = self.tokenizer_2(
                text,
                # padding='max_length',
                truncation=False,
                max_length=99999999,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)
            tokens_mask_2 = tokenizer_output.attention_mask.to(self.text_encoder_2.device)

        _, pooled_text_encoder_1_output = encode_clip(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-1,
            layer_skip=text_encoder_1_layer_skip,
            add_output=False,
            text_encoder_output=None,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_1_output,
            use_attention_mask=True,
        )
        if pooled_text_encoder_1_output is None:
            pooled_text_encoder_1_output = torch.zeros(
                size=(batch_size, 768),
                device=train_device,
                dtype=self.train_dtype.torch_dtype(),
            )

        with self.text_encoder_2_autocast_context:
            text_encoder_2_output = encode_t5(
                text_encoder=self.text_encoder_2,
                tokens=tokens_2,
                default_layer=-1,
                layer_skip=text_encoder_2_layer_skip,
                text_encoder_output=text_encoder_2_output,
                use_attention_mask=False,
                attention_mask=tokens_mask_2,
            )
            if text_encoder_2_output is None:
                text_encoder_2_output = torch.zeros(
                    size=(batch_size, 77, 4096),
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                )
                tokens_mask_2 = torch.zeros(
                    size=(batch_size, 1),
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                )

        if apply_attention_mask:
            text_encoder_2_output *= tokens_mask_2[:, :, None]

        # apply dropout
        if text_encoder_1_dropout_probability is not None:
            dropout_text_encoder_1_mask = (torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            pooled_text_encoder_1_output = pooled_text_encoder_1_output * dropout_text_encoder_1_mask[:, None]

        if text_encoder_2_dropout_probability is not None:
            dropout_text_encoder_2_mask = (torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask[:, None, None]

        return text_encoder_2_output, pooled_text_encoder_1_output

    def prepare_latent_image_ids(self, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def unpack_latents(self, latents, height, width):
        batch_size, num_patches, channels = latents.shape

        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents
