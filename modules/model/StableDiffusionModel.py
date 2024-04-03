from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionDepth2ImgPipeline, \
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import \
    rescale_noise_scheduler_to_zero_terminal_snr
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class StableDiffusionModelEmbedding:
    def __init__(
            self,
            text_encoder_vector: Tensor,
            prefix: str,
    ):
        token_count = text_encoder_vector.shape[0]

        self.text_encoder_vector = text_encoder_vector
        self.text_tokens = [f"<{prefix}_{i}>" for i in range(token_count)]


class StableDiffusionModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    image_depth_processor: DPTImageProcessor
    depth_estimator: DPTForDepthEstimation

    # autocast context
    autocast_context: torch.autocast | nullcontext

    train_dtype: DataType

    # persistent embedding training data
    all_text_encoder_original_token_embeds: Tensor
    text_encoder_untrainable_token_embeds_mask: list[bool]
    embeddings: list[StableDiffusionModelEmbedding] | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None

    sd_config: dict | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer: CLIPTokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder: CLIPTextModel | None = None,
            vae: AutoencoderKL | None = None,
            unet: UNet2DConditionModel | None = None,
            image_depth_processor: DPTImageProcessor | None = None,
            depth_estimator: DPTForDepthEstimation | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embeddings: list[StableDiffusionModelEmbedding] = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            sd_config: dict | None = None,
            model_spec: ModelSpec | None = None,
            train_config: TrainConfig | None = None,
    ):
        super(StableDiffusionModel, self).__init__(
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
        self.unet = unet
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

        self.autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32

        self.embeddings = embeddings if embeddings is not None else []
        self.text_encoder_lora = text_encoder_lora
        self.unet_lora = unet_lora
        self.sd_config = sd_config

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def depth_estimator_to(self, device: torch.device):
        if self.depth_estimator is not None:
            self.depth_estimator.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.depth_estimator_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self):
        self.vae.eval()
        if self.depth_estimator is not None:
            self.depth_estimator.eval()
        self.text_encoder.eval()
        self.unet.eval()

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

    def force_v_prediction(self):
        self.noise_scheduler.config.prediction_type = 'v_prediction'
        self.sd_config['model']['params']['parameterization'] = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        self.sd_config['model']['params']['parameterization'] = 'epsilon'

    def rescale_noise_scheduler_to_zero_terminal_snr(self):
        rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)
