from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionDepth2ImgPipeline, \
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, DiffusionPipeline
from torch import Tensor
from torch.optim import Optimizer
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionModelEmbedding:
    def __init__(self, name: str, vector: Tensor, token_count: int):
        self.name = name
        self.vector = vector
        self.token_count = token_count


class StableDiffusionModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    image_depth_processor: DPTImageProcessor
    depth_estimator: DPTForDepthEstimation

    # persistent training data
    embeddings: list[StableDiffusionModelEmbedding] | None
    text_encoder_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer: CLIPTokenizer,
            noise_scheduler: DDPMScheduler,
            text_encoder: CLIPTextModel,
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            image_depth_processor: DPTImageProcessor | None = None,
            depth_estimator: DPTForDepthEstimation | None = None,
            optimizer_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embeddings: list[StableDiffusionModelEmbedding] = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
    ):
        super(StableDiffusionModel, self).__init__(model_type, optimizer_state_dict, train_progress)

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

        self.embeddings = embeddings if embeddings is not None else []
        self.text_encoder_lora = text_encoder_lora
        self.unet_lora = unet_lora

    def create_pipeline(self) -> DiffusionPipeline:
        if self.model_type.has_depth_input():
            return StableDiffusionDepth2ImgPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                depth_estimator=self.depth_estimator,
                feature_extractor=self.image_depth_processor,
            )
        elif self.model_type.has_conditioning_image_input():
            return StableDiffusionInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        else:
            return StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
