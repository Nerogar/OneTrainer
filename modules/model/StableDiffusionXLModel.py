import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class StableDiffusionXLModelEmbedding:
    def __init__(
            self,
            text_encoder_1_vector: Tensor,
            text_encoder_2_vector: Tensor,
            prefix: str,
    ):
        token_count = text_encoder_1_vector.shape[0]

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector
        self.text_tokens = [f"< {prefix}_{i}>" for i in range(token_count)]


class StableDiffusionXLModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel

    # persistent embedding training data
    all_text_encoder_1_original_token_embeds: Tensor
    all_text_encoder_2_original_token_embeds: Tensor
    text_encoder_1_untrainable_token_embeds_mask: list[bool]
    text_encoder_2_untrainable_token_embeds_mask: list[bool]
    embeddings: list[StableDiffusionXLModelEmbedding] | None

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None

    sd_config: dict | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer_1: CLIPTokenizer | None = None,
            tokenizer_2: CLIPTokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder_1: CLIPTextModel | None = None,
            text_encoder_2: CLIPTextModel | None = None,
            vae: AutoencoderKL | None = None,
            unet: UNet2DConditionModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embeddings: list[StableDiffusionXLModelEmbedding] = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            sd_config: dict | None = None,
            model_spec: ModelSpec | None = None,
    ):
        super(StableDiffusionXLModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
        )

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.unet = unet

        self.embeddings = embeddings if embeddings is not None else []
        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.unet_lora = unet_lora
        self.sd_config = sd_config

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
