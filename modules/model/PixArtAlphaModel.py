from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DDIMScheduler, Transformer2DModel, \
    PixArtAlphaPipeline
from torch import Tensor
from transformers import T5Tokenizer, \
    T5EncoderModel

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class PixArtAlphaModelEmbedding:
    def __init__(
            self,
            text_encoder_vector: Tensor,
            prefix: str,
    ):
        token_count = text_encoder_vector.shape[0]

        self.text_encoder_vector = text_encoder_vector
        self.text_tokens = [f"<{prefix}_{i}>" for i in range(token_count)]


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
    transformer_autocast_context: torch.autocast | nullcontext
    vae_autocast_context: torch.autocast | nullcontext

    # persistent embedding training data
    all_text_encoder_original_token_embeds: Tensor
    text_encoder_untrainable_token_embeds_mask: list[bool]
    embeddings: list[PixArtAlphaModelEmbedding] | None

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
            embeddings: list[PixArtAlphaModelEmbedding] = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            model_spec: ModelSpec | None = None,
    ):
        super(PixArtAlphaModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
        )

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer

        self.autocast_context = nullcontext()
        self.text_encoder_autocast_context = nullcontext()
        self.transformer_autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        self.embeddings = embeddings if embeddings is not None else []
        self.text_encoder_lora = text_encoder_lora
        self.unet_lora = unet_lora

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        self.transformer.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return PixArtAlphaPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
        )
