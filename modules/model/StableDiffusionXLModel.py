from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, EulerDiscreteScheduler, \
    StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionXLModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: EulerDiscreteScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel

    # persistent training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer_1: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            noise_scheduler: EulerDiscreteScheduler,
            text_encoder_1: CLIPTextModel,
            text_encoder_2: CLIPTextModel,
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
    ):
        super(StableDiffusionXLModel, self).__init__(model_type, optimizer_state_dict, ema_state_dict, train_progress)

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.unet = unet

        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.unet_lora = unet_lora

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
