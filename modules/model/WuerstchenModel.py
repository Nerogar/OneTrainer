from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch
import torchvision
from torch import Tensor, nn

from diffusers import ConfigMixin, DDPMWuerstchenScheduler, DiffusionPipeline, ModelMixin, WuerstchenCombinedPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models import StableCascadeUNet
from diffusers.pipelines.stable_cascade import StableCascadeCombinedPipeline
from diffusers.pipelines.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt, WuerstchenPrior
from transformers import CLIPTextModel, CLIPTokenizer


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
            uuid: str,
            prior_text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.prior_text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=prior_text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )


class WuerstchenModel(BaseModel):
    # base model data
    decoder_tokenizer: CLIPTokenizer | None
    decoder_noise_scheduler: DDPMWuerstchenScheduler | None
    decoder_text_encoder: CLIPTextModel | None
    decoder_decoder: WuerstchenDiffNeXt | StableCascadeUNet | None
    decoder_vqgan: PaellaVQModel | None
    effnet_encoder: WuerstchenEfficientNetEncoder | None
    prior_tokenizer: CLIPTokenizer | None
    prior_text_encoder: CLIPTextModel | None
    prior_noise_scheduler: DDPMWuerstchenScheduler | None
    prior_prior: WuerstchenPrior | StableCascadeUNet | None

    # autocast context
    prior_autocast_context: torch.autocast | nullcontext
    effnet_encoder_autocast_context: torch.autocast | nullcontext

    prior_train_dtype: DataType
    effnet_encoder_train_dtype: DataType

    # persistent embedding training data
    embedding: WuerstchenModelEmbedding | None
    additional_embeddings: list[WuerstchenModelEmbedding] | None
    prior_embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    prior_text_encoder_lora: LoRAModuleWrapper | None
    prior_prior_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.decoder_tokenizer = None
        self.decoder_noise_scheduler = None
        self.decoder_text_encoder = None
        self.decoder_decoder = None
        self.decoder_vqgan = None
        self.effnet_encoder = None
        self.prior_tokenizer = None
        self.prior_text_encoder = None
        self.prior_noise_scheduler = None
        self.prior_prior = None

        self.prior_autocast_context = nullcontext()
        self.effnet_encoder_autocast_context = nullcontext()

        self.prior_train_dtype = DataType.FLOAT_32
        self.effnet_encoder_train_dtype = DataType.FLOAT_32

        self.embedding = None
        self.additional_embeddings = []
        self.prior_embedding_wrapper = None

        self.prior_text_encoder_lora = None
        self.prior_prior_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.prior_text_encoder_lora,
            self.prior_prior_lora,
        ] if a is not None]

    def all_embeddings(self) -> list[WuerstchenModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_prior_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.prior_text_encoder_embedding] if self.embedding is not None else [])

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
                prior_tokenizer=self.prior_tokenizer,
                prior_text_encoder=self.prior_text_encoder,
                prior_prior=self.prior_prior,
                prior_scheduler=self.prior_noise_scheduler,
            )
        raise NotImplementedError

    def add_prior_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_prior_text_encoder_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
            pooled_text_encoder_output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            tokenizer_output = self.prior_tokenizer(
                self.add_prior_text_encoder_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.prior_text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.prior_text_encoder.device)

        if text_encoder_output is None:
            text_encoder_output = self.prior_text_encoder(
                tokens,
                attention_mask=tokens_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if self.model_type.is_wuerstchen_v2():
                final_layer_norm = self.prior_text_encoder.text_model.final_layer_norm
                pooled_text_encoder_output = None
                text_encoder_output = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
                )
            if self.model_type.is_stable_cascade():
                pooled_text_encoder_output = text_encoder_output.text_embeds.unsqueeze(1)
                text_encoder_output = text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
        else:
            if self.model_type.is_stable_cascade():
                pooled_text_encoder_output = pooled_text_encoder_output.unsqueeze(1)

        text_encoder_output = self._apply_output_embeddings(
            self.all_prior_text_encoder_embeddings(),
            self.prior_tokenizer,
            tokens,
            text_encoder_output,
        )

        # apply dropout
        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()

            if self.model_type.is_wuerstchen_v2():
                text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]
            if self.model_type.is_stable_cascade():
                pooled_text_encoder_output = pooled_text_encoder_output * dropout_text_encoder_mask[:, None, None]
                text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output, pooled_text_encoder_output
