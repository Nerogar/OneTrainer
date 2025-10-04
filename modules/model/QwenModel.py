import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLQwenImage,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

DEFAULT_PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_PROMPT_TEMPLATE_CROP_START = 34
PROMPT_MAX_LENGTH = 512

class QwenModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen2_5_VLForConditionalGeneration | None
    vae: AutoencoderKLQwenImage | None
    transformer: QwenImageTransformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
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

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device): #TODO share more code between models
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

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
        return QwenImagePipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            text = [DEFAULT_PROMPT_TEMPLATE.format(t) for t in text]
            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH + DEFAULT_PROMPT_TEMPLATE_CROP_START,
                padding='longest',
                truncation=True,
                return_tensors="pt"
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                text_encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    output_hidden_states=True,
                    return_dict=True,
                )
                text_encoder_output = text_encoder_output.hidden_states[-1]
                tokens_mask = tokens_mask[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:]

                #TODO diffusers splits the prompts and stacks them again. Why?
                #https://github.com/huggingface/diffusers/blob/fc337d585309c4b032e8d0180bea683007219df1/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L211
                #https://github.com/huggingface/diffusers/issues/12295

                #set masked state to 0 should not make a difference, but this seems to be the only effect of the diffusers code links above
                text_encoder_output = text_encoder_output[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:,:] * tokens_mask.unsqueeze(-1)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError #https://github.com/Nerogar/OneTrainer/issues/957

        #prune tokens that are masked in all batch samples
        #this is still necessary even though we are using 'longest' padding, because cached
        #encoder outputs by MGDS are always PROMPT_MAX_LENGTH
        #this is good for efficiency, but also FIXME currently required by the diffusers pipeline:
        #https://github.com/huggingface/diffusers/issues/12344
        seq_lengths = tokens_mask.sum(dim=1)
        max_seq_length = seq_lengths.max().item()
        text_encoder_output = text_encoder_output[:, :max_seq_length, :]
        bool_attention_mask = tokens_mask[:, :max_seq_length].bool()

        return (text_encoder_output, bool_attention_mask)

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = latents.shape
        assert frames == 1

        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

        return latents

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, _, channels = latents.shape

        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height * 2, width * 2)

        return latents

    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents / latents_std + latents_mean

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
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
