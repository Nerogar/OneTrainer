import torchvision
from diffusers import DiffusionPipeline, DDPMWuerstchenScheduler, WuerstchenCombinedPipeline, ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.pipelines.wuerstchen import WuerstchenDiffNeXt, PaellaVQModel, WuerstchenPrior
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class WuerstchenEfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            c_latent: int = 16,
            c_cond: int = 1280,
            effnet: str = "efficientnet_v2_s",
    ):
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = torchvision.models.efficientnet_v2_s().features
        else:
            self.backbone = torchvision.models.efficientnet_v2_l().features

        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))


class WuerstchenModel(BaseModel):
    # base model data
    model_type: ModelType
    decoder_tokenizer: CLIPTokenizer
    decoder_noise_scheduler: DDPMWuerstchenScheduler
    decoder_text_encoder: CLIPTextModel
    decoder_decoder: WuerstchenDiffNeXt
    decoder_vqgan: PaellaVQModel
    effnet_encoder: WuerstchenEfficientNetEncoder
    prior_tokenizer: CLIPTokenizer
    prior_text_encoder: CLIPTextModel
    prior_noise_scheduler: DDPMWuerstchenScheduler
    prior_prior: WuerstchenPrior

    # persistent training data
    prior_text_encoder_lora: LoRAModuleWrapper | None
    prior_prior_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
            decoder_tokenizer: CLIPTokenizer | None = None,
            decoder_noise_scheduler: DDPMWuerstchenScheduler | None = None,
            decoder_text_encoder: CLIPTextModel | None = None,
            decoder_decoder: WuerstchenDiffNeXt | None = None,
            decoder_vqgan: PaellaVQModel | None = None,
            effnet_encoder: WuerstchenEfficientNetEncoder | None = None,
            prior_tokenizer: CLIPTokenizer | None = None,
            prior_text_encoder: CLIPTextModel | None = None,
            prior_noise_scheduler: DDPMWuerstchenScheduler | None = None,
            prior_prior: WuerstchenPrior | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            prior_text_encoder_lora: LoRAModuleWrapper | None = None,
            prior_prior_lora: LoRAModuleWrapper | None = None,
            model_spec: ModelSpec | None = None,
    ):
        super(WuerstchenModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
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
        self.prior_text_encoder_lora = prior_text_encoder_lora
        self.prior_prior_lora = prior_prior_lora

    def create_pipeline(self) -> DiffusionPipeline:
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
