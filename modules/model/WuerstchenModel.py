from contextlib import nullcontext

import torch
import torchvision
from diffusers import DiffusionPipeline, DDPMWuerstchenScheduler, WuerstchenCombinedPipeline, ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.pipelines.stable_cascade import StableCascadeUnet, StableCascadeCombinedPipeline
from diffusers.pipelines.wuerstchen import WuerstchenDiffNeXt, PaellaVQModel, WuerstchenPrior
from torch import nn, Tensor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class WuerstchenEfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            c_latent: int = 16,
            c_cond: int = 1280,
            effnet: str = "efficientnet_v2_s",
            affine_batch_norm: bool = True,
    ):
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = torchvision.models.efficientnet_v2_s().features
        else:
            self.backbone = torchvision.models.efficientnet_v2_l().features

        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=affine_batch_norm),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))


class WuerstchenModelEmbedding:
    def __init__(
            self,
            prior_text_encoder_vector: Tensor,
            prefix: str,
    ):
        token_count = prior_text_encoder_vector.shape[0]

        self.prior_text_encoder_vector = prior_text_encoder_vector
        self.text_tokens = [f"<{prefix}_{i}>" for i in range(token_count)]


class WuerstchenModel(BaseModel):
    # base model data
    model_type: ModelType
    decoder_tokenizer: CLIPTokenizer
    decoder_noise_scheduler: DDPMWuerstchenScheduler
    decoder_text_encoder: CLIPTextModel
    decoder_decoder: WuerstchenDiffNeXt | StableCascadeUnet
    decoder_vqgan: PaellaVQModel
    effnet_encoder: WuerstchenEfficientNetEncoder
    prior_tokenizer: CLIPTokenizer
    prior_text_encoder: CLIPTextModel
    prior_noise_scheduler: DDPMWuerstchenScheduler
    prior_prior: WuerstchenPrior | StableCascadeUnet

    # autocast context
    autocast_context: torch.autocast | nullcontext

    train_dtype: DataType

    # persistent embedding training data
    all_prior_text_encoder_original_token_embeds: Tensor
    prior_text_encoder_untrainable_token_embeds_mask: list[bool]
    embeddings: list[WuerstchenModelEmbedding] | None

    # persistent lora training data
    prior_text_encoder_lora: LoRAModuleWrapper | None
    prior_prior_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
            decoder_tokenizer: CLIPTokenizer | None = None,
            decoder_noise_scheduler: DDPMWuerstchenScheduler | None = None,
            decoder_text_encoder: CLIPTextModel | None = None,
            decoder_decoder: WuerstchenDiffNeXt | StableCascadeUnet | None = None,
            decoder_vqgan: PaellaVQModel | None = None,
            effnet_encoder: WuerstchenEfficientNetEncoder | None = None,
            prior_tokenizer: CLIPTokenizer | None = None,
            prior_text_encoder: CLIPTextModel | CLIPTextModelWithProjection | None = None,
            prior_noise_scheduler: DDPMWuerstchenScheduler | None = None,
            prior_prior: WuerstchenPrior | StableCascadeUnet | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embeddings: list[WuerstchenModelEmbedding] | None = None,
            prior_text_encoder_lora: LoRAModuleWrapper | None = None,
            prior_prior_lora: LoRAModuleWrapper | None = None,
            model_spec: ModelSpec | None = None,
            train_config: TrainConfig | None = None,
    ):
        super(WuerstchenModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
            train_config=train_config,
        )

        self.decoder_tokenizer = decoder_tokenizer
        self.decoder_noise_scheduler = decoder_noise_scheduler
        self.decoder_text_encoder = decoder_text_encoder
        self.decoder_decoder = decoder_decoder
        self.decoder_vqgan = decoder_vqgan
        self.effnet_encoder = effnet_encoder
        self.prior_tokenizer = prior_tokenizer
        self.prior_text_encoder = prior_text_encoder
        self.prior_noise_scheduler = prior_noise_scheduler
        self.prior_prior = prior_prior

        self.autocast_context = nullcontext()
        self.prior_autocast_context = nullcontext()
        self.effnet_encoder_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.prior_train_dtype = DataType.FLOAT_32
        self.effnet_encoder_train_dtype = DataType.FLOAT_32

        self.embeddings = embeddings if embeddings is not None else []
        self.prior_text_encoder_lora = prior_text_encoder_lora
        self.prior_prior_lora = prior_prior_lora

    def decoder_text_encoder_to(self, device: torch.device):
        self.decoder_text_encoder.to(device=device)

    def decoder_decoder_to(self, device: torch.device):
        self.decoder_decoder.to(device=device)

    def decoder_vqgan_to(self, device: torch.device):
        self.decoder_vqgan.to(device=device)

    def effnet_encoder_to(self, device: torch.device):
        self.effnet_encoder.to(device=device)

    def prior_text_encoder_to(self, device: torch.device):
        self.prior_text_encoder.to(device=device)

        if self.prior_text_encoder_lora is not None:
            self.prior_text_encoder_lora.to(device)

    def prior_prior_to(self, device: torch.device):
        self.prior_prior.to(device=device)

        if self.prior_prior_lora is not None:
            self.prior_prior_lora.to(device)

    def to(self, device: torch.device):
        if self.model_type.is_wuerstchen_v2():
            self.decoder_text_encoder_to(device)
        self.decoder_decoder_to(device)
        self.decoder_vqgan_to(device)
        self.effnet_encoder_to(device)
        self.prior_text_encoder_to(device)
        self.prior_prior_to(device)

    def eval(self):
        if self.model_type.is_wuerstchen_v2():
            self.decoder_text_encoder.eval()
        self.decoder_decoder.eval()
        self.decoder_vqgan.eval()
        self.effnet_encoder.eval()
        self.prior_text_encoder.eval()
        self.prior_prior.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        if self.model_type.is_wuerstchen_v2():
            return WuerstchenCombinedPipeline(
                tokenizer=self.decoder_tokenizer,
                text_encoder=self.decoder_text_encoder,
                decoder=self.decoder_decoder,
                scheduler=self.decoder_noise_scheduler,
                vqgan=self.decoder_vqgan,
                prior_tokenizer=self.prior_tokenizer,
                prior_text_encoder=self.prior_text_encoder,
                prior_prior=self.prior_prior,
                prior_scheduler=self.prior_noise_scheduler,
            )
        elif self.model_type.is_stable_cascade():
            return StableCascadeCombinedPipeline(
                tokenizer=self.prior_tokenizer,
                text_encoder=self.prior_text_encoder,
                decoder=self.decoder_decoder,
                scheduler=self.decoder_noise_scheduler,
                vqgan=self.decoder_vqgan,
                prior_prior=self.prior_prior,
                prior_scheduler=self.prior_noise_scheduler,
            )
