import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.OnDemandModule import OnDemandModule
from modules.util.torch_util import torch_gc

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import PreTrainedTokenizerFast

from lens.pipeline import LensPipeline
from lens.text_encoder import LensGptOssEncoder
from lens.transformer import LensTransformer2DModel

# Chat template constants, matching lens/pipeline.py
CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
PROMPT_TEMPLATE_CROP_START = 97    # tokens consumed by the chat template prefix
PROMPT_MAX_LENGTH = 512  # caption token budget (chat template tokens are added on top)


def make_lens_conversation(caption: str) -> list[dict]:
    return [
        {"role": "system", "content": CHAT_SYSTEM, "thinking": None},
        {"role": "user", "content": caption, "thinking": None},
        {"role": "assistant", "thinking": CHAT_ASSISTANT_THINKING, "content": ""},
    ]


class LensModel(BaseModel):
    # base model data
    tokenizer: PreTrainedTokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: LensGptOssEncoder | OnDemandModule | None
    text_encoder_hidden_size: int | None  # cached so encode_text() works after encoder is deleted
    vae: AutoencoderKLFlux2 | None
    transformer: LensTransformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

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
        self.text_encoder_hidden_size = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

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
        # only moves a resident/materialized encoder. The on-demand proxy's to() is a no-op;
        # its placement is owned by materialize_text_encoder / release_text_encoder.
        if self.text_encoder is not None:
            self.text_encoder.to(device=device)

    # Lens always loads on demand (see TrainConfig.text_encoder_on_demand): the GPT-OSS encoder is
    # MXFP4-quantized and cannot be parked on the CPU temp device, so it is rebuilt straight onto the
    # accelerator when needed and discarded afterwards. MXFP4 quantization happens inside
    # from_pretrained (the loader lambda), so no quantize_layers call is needed here.
    def materialize_text_encoder(self, device: torch.device):
        if isinstance(self.text_encoder, OnDemandModule):
            self.text_encoder.materialize()
            self.text_encoder.inner.to(device)
        else:
            self.text_encoder_to(device)  # resident: just move

    # load an on-demand encoder for saving. resident encoders already hold their weights.
    def materialize_text_encoder_for_save(self):
        if isinstance(self.text_encoder, OnDemandModule):
            self.text_encoder.materialize()

    def release_text_encoder(self):
        if isinstance(self.text_encoder, OnDemandModule):
            self.text_encoder.discard()  # free the weights
            torch_gc()
        else:
            self.text_encoder_to(torch.device(self.train_config.temp_device))  # resident: park on temp

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

    # park resident components on the temp device and discard the on-demand text encoder
    def release(self):
        temp_device = torch.device(self.train_config.temp_device)
        self.vae_to(temp_device)
        self.release_text_encoder()
        self.transformer_to(temp_device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    # the real module behind the (possibly on-demand) text_encoder: the inner module when
    # materialized, None when discarded, or the resident module. The pipeline and save path
    # need the unwrapped module, not the proxy.
    def _resolved_text_encoder(self) -> torch.nn.Module | None:
        if isinstance(self.text_encoder, OnDemandModule):
            return self.text_encoder.inner
        return self.text_encoder

    def create_pipeline(self) -> DiffusionPipeline:
        return LensPipeline(
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self._resolved_text_encoder(),
            tokenizer=self.tokenizer,
            transformer=self.transformer,
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
            text_encoder_output: list[Tensor] | None = None,
    ) -> tuple[list[Tensor] | None, Tensor | None]:

        # the on-demand proxy is always truthy, so resolve the real (materialized) encoder here and
        # gate the fresh-encode path on it being present rather than on `self.text_encoder`.
        text_encoder = self._resolved_text_encoder()

        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            rendered = []
            for prompt in text:
                t = self.tokenizer.apply_chat_template(
                    make_lens_conversation(prompt), tokenize=False, add_generation_prompt=False
                )
                rendered.append(t.split("<|return|>")[0])

            tokenizer_output = self.tokenizer(
                rendered,
                max_length=PROMPT_MAX_LENGTH + PROMPT_TEMPLATE_CROP_START,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            tokens = tokenizer_output.input_ids.to(text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(text_encoder.device)

        if text_encoder_output is None and text_encoder is not None:
            with self.text_encoder_autocast_context:
                # encode_layers() is used instead of the standard output_hidden_states=True API
                # because transformers' @capture_outputs decorator applies tie_last_hidden_states,
                # which silently replaces hidden_states[-1] with the norm-applied final output.
                # GPT-OSS has 24 layers and selected_layer_index ends at layer 23 (the last),
                # so output_hidden_states=True returns the normed version for that layer while
                # encode_layers() correctly returns the pre-norm hidden state.
                layer_outputs = text_encoder.encode_layers(tokens, tokens_mask)
                if tokens.shape[1] > PROMPT_TEMPLATE_CROP_START:
                    text_encoder_output = [feat[:, PROMPT_TEMPLATE_CROP_START:, :].contiguous() for feat in layer_outputs]
                    tokens_mask = tokens_mask[:, PROMPT_TEMPLATE_CROP_START:]
                else:
                    #TODO can this ever happen? max_length=PROMPT_MAX_LENGTH+PROMPT_TEMPLATE_CROP_START,
                    #so the tokenizer output should always be longer than PROMPT_TEMPLATE_CROP_START.
                    #upstream pipeline has the same guard, presumably as a safety net.
                    zero_shape = (tokens.shape[0], 0, layer_outputs[0].shape[-1])
                    text_encoder_output = [layer_outputs[0].new_zeros(zero_shape) for _ in layer_outputs]
                    tokens_mask = torch.zeros((tokens.shape[0], 0), dtype=torch.bool, device=tokens.device)

        elif text_encoder_output is not None:
            # Cached: EncodeLensText concatenates layers along dim=-1; split back into per-layer list.
            hidden_dim = self.text_encoder_hidden_size
            text_encoder_output = list(text_encoder_output.split(hidden_dim, dim=-1))

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # https://github.com/Nerogar/OneTrainer/issues/957

        #prune tokens that are masked in all batch samples:
        seq_lengths = tokens_mask.sum(dim=1)
        max_seq_length = int(seq_lengths.max().item())

        #pad to 16 because attention processors and/or torch.compile can have issues with uneven sequence lengths, but only pad if an attention mask has to be used anyway:
        if max_seq_length % 16 > 0 and (seq_lengths != max_seq_length).any():
            max_seq_length += (16 - max_seq_length % 16)

        text_encoder_output = [feat[:, :max_seq_length, :] for feat in text_encoder_output]
        tokens_mask = tokens_mask[:, :max_seq_length].bool()

        return text_encoder_output, tokens_mask

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

    #packing and unpacking on patchified latents
    @staticmethod
    def pack_latents(latents) -> Tensor:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, seq_len, num_channels = latents.shape
        return latents.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

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
