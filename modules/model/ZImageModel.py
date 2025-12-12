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
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

PROMPT_MAX_LENGTH = 512

def format_input(text: str):
    return [
        {"role": "user", "content": text},
    ]


class ZImageModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3ForCausalLM | None
    vae: AutoencoderKL | None
    transformer: ZImageTransformer2DModel | None

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

        self.text_encoder_train_dtype = DataType.FLOAT_32 #TODO

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

    def text_encoder_to(self, device: torch.device): #TODO share more code between models
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
        return ZImagePipeline(
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
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            for i, prompt_item in enumerate(text):
                messages = format_input(prompt_item)
                prompt_item = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                text[i] = prompt_item

            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding='max_length',
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
                text_encoder_output = text_encoder_output.hidden_states[-2]

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError #https://github.com/Nerogar/OneTrainer/issues/957

        embeddings_list = []
        bool_attention_mask = tokens_mask.bool()
        embeddings_list = [sample[bool_attention_mask[i]] for i, sample in enumerate(text_encoder_output)]
        return embeddings_list

    def scale_latents(self, latents: Tensor) -> Tensor:
        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def unscale_latents(self, latents: Tensor) -> Tensor:
        return latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

    '''def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)'''
