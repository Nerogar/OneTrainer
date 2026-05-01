import math
from contextlib import nullcontext
from random import Random

PROMPT_MAX_LENGTH = 512
HIDDEN_STATES_LAYER = -2

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    ErnieImagePipeline,
    ErnieImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, Mistral3Model


class ErnieModel(BaseModel):
    # base model data
    tokenizer: AutoTokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Mistral3Model | None
    vae: AutoencoderKLFlux2 | None
    transformer: ErnieImageTransformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

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
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return ErnieImagePipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokenizer_output = self.tokenizer(
                text,
                padding='max_length',
                max_length=PROMPT_MAX_LENGTH,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            #this is different from the diffusers pipeline.
            #they accumulate the embeddings for each sample without padding or an attention mask. we use padded with an attention mask, because the dataloader pipeline does the same
            #It has been tested that this is identical.
            with self.text_encoder_autocast_context:
                output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                text_encoder_output = output.hidden_states[HIDDEN_STATES_LAYER]  # [B, T, H]

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # needs empty-caption conditioning, not zero-out

        text_lengths = tokens_mask.sum(dim=1).long()
        text_encoder_output = text_encoder_output[:, :text_lengths.max().item(), :]
        return text_encoder_output, text_lengths

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    def scale_latents(self, latents: Tensor) -> Tensor:
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * std + mean

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
