from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import (
    rescale_noise_scheduler_to_zero_terminal_snr,
)
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, DPTForDepthEstimation, DPTImageProcessor


class StableDiffusionModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )


class StableDiffusionModel(BaseModel):
    # base model data
    tokenizer: CLIPTokenizer | None
    noise_scheduler: DDIMScheduler | None
    text_encoder: CLIPTextModel | None
    vae: AutoencoderKL | None
    unet: UNet2DConditionModel | None
    image_depth_processor: DPTImageProcessor | None
    depth_estimator: DPTForDepthEstimation | None

    # persistent embedding training data
    embedding: StableDiffusionModelEmbedding | None
    additional_embeddings: list[StableDiffusionModelEmbedding] | None
    embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.image_depth_processor = None
        self.depth_estimator = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper = None

        self.text_encoder_lora = None
        self.unet_lora = None
        self.lora_state_dict = None

        self.sd_config = None
        self.sd_config_filename = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.unet_lora,
        ] if a is not None]

    def all_embeddings(self) -> list[StableDiffusionModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_embedding] if self.embedding is not None else [])

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

    def add_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ):
        if tokens is None:
            tokenizer_output = self.tokenizer(
                self.add_text_encoder_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=self.text_encoder.config.max_position_embeddings,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)

        text_encoder_output, _ = encode_clip(
            text_encoder=self.text_encoder,
            tokens=tokens,
            default_layer=-1,
            layer_skip=text_encoder_layer_skip,
            text_encoder_output=text_encoder_output,
            add_pooled_output=False,
            use_attention_mask=False,
            add_layer_norm=True,
        )

        text_encoder_output = self._apply_output_embeddings(
            self.all_text_encoder_embeddings(),
            self.tokenizer,
            tokens,
            text_encoder_output,
        )

        # apply dropout
        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output
