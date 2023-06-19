from diffusers import UNet2DConditionModel, DDIMScheduler, PriorTransformer, VQModel, \
    KandinskyPriorPipeline, KandinskyPipeline, UnCLIPScheduler
from diffusers.pipelines.kandinsky import MultilingualCLIP
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, \
    CLIPImageProcessor, XLMRobertaTokenizer, XLMRobertaTokenizerFast

from modules.model.BaseModel import BaseModel
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
    noise_scheduler: DDIMScheduler
    movq: VQModel

    def __init__(
            self,
            model_type: ModelType,

            # prior
            prior_tokenizer: CLIPTokenizer,
            prior_text_encoder: CLIPTextModelWithProjection,
            prior_image_encoder: CLIPVisionModelWithProjection,
            prior_prior: PriorTransformer,
            prior_noise_scheduler: UnCLIPScheduler,
            prior_image_processor: CLIPImageProcessor,

            # diffusion model
            tokenizer: XLMRobertaTokenizerFast,
            text_encoder: MultilingualCLIP,
            unet: UNet2DConditionModel,
            noise_scheduler: DDIMScheduler,
            movq: VQModel,

            optimizer_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
    ):
        super(KandinskyModel, self).__init__(model_type, optimizer_state_dict, train_progress)

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
