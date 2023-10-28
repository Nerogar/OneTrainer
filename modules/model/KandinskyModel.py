import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, PriorTransformer, VQModel, \
    KandinskyPriorPipeline, KandinskyPipeline, UnCLIPScheduler
from diffusers.pipelines.kandinsky import MultilingualCLIP
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, \
    CLIPImageProcessor, XLMRobertaTokenizerFast

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class KandinskyModel(BaseModel):
    # base model data
    model_type: ModelType

    # prior
    prior_tokenizer: CLIPTokenizer
    prior_text_encoder: CLIPTextModelWithProjection
    prior_image_encoder: CLIPVisionModelWithProjection
    prior_prior: PriorTransformer
    prior_noise_scheduler: UnCLIPScheduler
    prior_image_processor: CLIPImageProcessor

    # diffusion model
    tokenizer: XLMRobertaTokenizerFast
    text_encoder: MultilingualCLIP
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    movq: VQModel

    # persistent training data
    unet_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,

            # prior
            prior_tokenizer: CLIPTokenizer | None = None,
            prior_text_encoder: CLIPTextModelWithProjection | None = None,
            prior_image_encoder: CLIPVisionModelWithProjection | None = None,
            prior_prior: PriorTransformer | None = None,
            prior_noise_scheduler: UnCLIPScheduler | None = None,
            prior_image_processor: CLIPImageProcessor | None = None,

            # diffusion model
            tokenizer: XLMRobertaTokenizerFast | None = None,
            text_encoder: MultilingualCLIP | None = None,
            unet: UNet2DConditionModel | None = None,
            noise_scheduler: DDPMScheduler | None = None,
            movq: VQModel | None = None,

            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,

            unet_lora: LoRAModuleWrapper | None = None,
    ):
        super(KandinskyModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=None,
        )

        # prior
        self.prior_tokenizer = prior_tokenizer
        self.prior_text_encoder = prior_text_encoder
        self.prior_image_encoder = prior_image_encoder
        self.prior_prior = prior_prior
        self.prior_noise_scheduler = prior_noise_scheduler
        self.prior_image_processor = prior_image_processor

        # diffusion model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.movq = movq

        self.unet_lora = unet_lora

    def prior_text_encoder_to(self, device: torch.device):
        self.prior_text_encoder.to(device)

    def prior_image_encoder_to(self, device: torch.device):
        self.prior_image_encoder.to(device)

    def prior_prior_to(self, device: torch.device):
        self.prior_prior.to(device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def movq_to(self, device: torch.device):
        self.movq.to(device)

    def to(self, device: torch.device):
        self.prior_text_encoder_to(device)
        self.prior_text_encoder_to(device)
        self.prior_image_encoder_to(device)
        self.prior_prior_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)
        self.movq_to(device)

    def eval(self):
        self.prior_text_encoder.eval()
        self.prior_text_encoder.eval()
        self.prior_image_encoder.eval()
        self.prior_prior.eval()
        self.text_encoder.eval()
        self.unet.eval()
        self.movq.eval()

    def create_prior_pipeline(self) -> KandinskyPriorPipeline:
        return KandinskyPriorPipeline(
            tokenizer=self.prior_tokenizer,
            text_encoder=self.prior_text_encoder,
            image_encoder=self.prior_image_encoder,
            prior=self.prior_prior,
            scheduler=self.prior_noise_scheduler,
            image_processor=self.prior_image_processor,
        )

    def create_diffusion_pipeline(self) -> KandinskyPipeline:
        return KandinskyPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            movq=self.movq,
        )
