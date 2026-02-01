import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert_util import qkv_fusion, swap_chunks
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)
from diffusers.pipelines.flux2.pipeline_flux2 import format_input as mistral_format_input
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor, Qwen2Tokenizer, Qwen3ForCausalLM

MISTRAL_SYSTEM_MESSAGE = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."
MISTRAL_HIDDEN_STATES_LAYERS = [10, 20, 30]
QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]

def qwen3_format_input(text: str):
    return [
        {"role": "user", "content": text},
    ]


def diffusers_to_original(qkv_fusion):
    return [
        ("context_embedder", "txt_in"),
        ("x_embedder",       "img_in"),
        ("time_guidance_embed.timestep_embedder", "time_in", [
            ("linear_1", "in_layer"),
            ("linear_2", "out_layer"),
        ]),
        ("time_guidance_embed.guidance_embedder", "guidance_in", [
            ("linear_1", "in_layer"),
            ("linear_2", "out_layer"),
        ]),
        ("double_stream_modulation_img.linear", "double_stream_modulation_img.lin"),
        ("double_stream_modulation_txt.linear", "double_stream_modulation_txt.lin"),
        ("single_stream_modulation.linear",     "single_stream_modulation.lin"),
        ("proj_out",                            "final_layer.linear"),
        ("norm_out.linear", "final_layer.adaLN_modulation.1", swap_chunks, swap_chunks),
        ("transformer_blocks.{i}", "double_blocks.{i}",
            qkv_fusion("attn.to_q", "attn.to_k", "attn.to_v", "img_attn.qkv") + \
            qkv_fusion("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "txt_attn.qkv") + [
            ("attn.norm_k.weight",       "img_attn.norm.key_norm.scale"),
            ("attn.norm_q.weight",       "img_attn.norm.query_norm.scale"),
            ("attn.to_out.0",            "img_attn.proj"),
            ("ff.linear_in",             "img_mlp.0"),
            ("ff.linear_out",            "img_mlp.2"),
            ("attn.norm_added_k.weight", "txt_attn.norm.key_norm.scale"),
            ("attn.norm_added_q.weight", "txt_attn.norm.query_norm.scale"),
            ("attn.to_add_out",          "txt_attn.proj"),
            ("ff_context.linear_in",     "txt_mlp.0"),
            ("ff_context.linear_out",    "txt_mlp.2"),
        ]),
        ("single_transformer_blocks.{i}", "single_blocks.{i}", [
            ("attn.to_qkv_mlp_proj", "linear1"),
            ("attn.to_out",          "linear2"),
            ("attn.norm_k.weight",   "norm.key_norm.scale"),
            ("attn.norm_q.weight",   "norm.query_norm.scale"),
        ]),
    ]

diffusers_checkpoint_to_original = diffusers_to_original(qkv_fusion)

class Flux2Model(BaseModel):
    # base model data
    tokenizer: PixtralProcessor | Qwen2Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Mistral3ForConditionalGeneration | Qwen3ForCausalLM | None
    vae: AutoencoderKLFlux2 | None
    transformer: Flux2Transformer2DModel | None

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
        klass = Flux2Pipeline if self.is_dev() else Flux2KleinPipeline
        return klass(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1, #TODO unused
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_sequence_length: int | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            if self.is_dev():
                messages = mistral_format_input(prompts=text, system_message=MISTRAL_SYSTEM_MESSAGE)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                tokenizer_output = self.tokenizer(
                    text,
                    max_length=text_encoder_sequence_length, #max length is including system message
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
            else: #Flux2.Klein
                for i, prompt_item in enumerate(text):
                    messages = qwen3_format_input(prompt_item)
                    prompt_item = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    text[i] = prompt_item

                tokenizer_output = self.tokenizer(
                    text,
                    max_length=text_encoder_sequence_length,
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
                    use_cache=False,
                )
                text_encoder_output = torch.cat([text_encoder_output.hidden_states[k]
                                                   for k in (MISTRAL_HIDDEN_STATES_LAYERS if self.is_dev() else QWEN3_HIDDEN_STATES_LAYERS)], dim=2)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError #https://github.com/Nerogar/OneTrainer/issues/957

        return text_encoder_output

    def is_dev(self) -> bool:
        return isinstance(self.tokenizer, PixtralProcessor)

    def is_klein(self) -> bool:
        return not self.is_dev()

    #code adapted from https://github.com/huggingface/diffusers/blob/c8656ed73c638e51fc2e777a5fd355d69fa5220f/src/diffusers/pipelines/flux2/pipeline_flux2.py
    @staticmethod
    def prepare_latent_image_ids(latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = latents.shape

        t = torch.arange(1, device=latents.device)
        h = torch.arange(height, device=latents.device)
        w = torch.arange(width, device=latents.device)
        l_ = torch.arange(1, device=latents.device)

        latent_ids = torch.cartesian_prod(t, h, w, l_)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    #packing and unpacking on patchified latents
    @staticmethod
    def pack_latents(latents) -> Tensor:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, seq_len, num_channels = latents.shape
        return latents.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

    #inference code uses empirical mu. But that code cannot be used for training because it depends on num of inference steps, and is likely too high for training
    #the dynamic shifting parameters of the noise schedulers are probably just the default values (taken from Flux1) and not applicable - but the best values we have:
    #unpatchified width and height
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

    @staticmethod
    def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        out_ids = []

        for _ in range(B): #TODO why iterate? can text ids have different length? according to diffusers and original inference code: no
            t = torch.arange(1, device=x.device)
            h = torch.arange(1, device=x.device)
            w = torch.arange(1, device=x.device)
            l_ = torch.arange(L, device=x.device)

            coords = torch.cartesian_prod(t, h, w, l_)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    #scaling on patchified latents
    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return (latents - latents_bn_mean) / latents_bn_std


    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return latents * latents_bn_std + latents_bn_mean
