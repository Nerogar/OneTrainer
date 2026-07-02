from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import (
    rescale_noise_scheduler_to_zero_terminal_snr,
)
from modules.util.enum.ModelFormat import ModelFormat
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

# diffusers -> original/sgm resnet-block leaf renames (full-weight; the norm leaves carry no LoRA, so
# those rules are inert on a LoRA state dict). skip_connection only fires when present.
_RESNET = [
    ("norm1", "in_layers.0"),
    ("conv1", "in_layers.2"),
    ("time_emb_proj", "emb_layers.1"),
    ("norm2", "out_layers.0"),
    ("conv2", "out_layers.3"),
    ("conv_shortcut", "skip_connection"),
]


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
    orig_tokenizer: CLIPTokenizer | None
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
        self.orig_tokenizer = None
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

    def diffusers_to_original(self) -> list | None:
        # SD1.5/2.x UNet diffusers -> original/sgm key map, convert()-native (bare sgm names, no top prefix).
        # SD has NO qkv fusion and NO add_embedding, so this is a pure key rename. Spatial-transformer
        # (attention) blocks have IDENTICAL leaf names in both namespaces -- only the block addressing differs
        # -- so each is a single 2-tuple rename; only the resnet blocks (_RESNET) need a leaf map. The
        # diffusers (block, sub-index) -> flat sgm input/output_blocks index is arithmetic 3*i+j(+1) that
        # parse-substitution can't express, so per-block rules are generated in a loop. SD is a 4-level UNet:
        # down attentions in blocks 0-2, up attentions in blocks 1-3.
        rules = [
            ("conv_in", "input_blocks.0.0"),
            ("time_embedding.linear_1", "time_embed.0"),
            ("time_embedding.linear_2", "time_embed.2"),
            ("conv_norm_out", "out.0"),
            ("conv_out", "out.2"),
        ]

        # down blocks: input_blocks index = 3*i + j + 1; attentions in blocks 0,1,2; downsamplers in 0,1,2.
        for i in range(4):
            for j in range(2):
                sgm = 3 * i + j + 1
                rules.append((f"down_blocks.{i}.resnets.{j}", f"input_blocks.{sgm}.0", _RESNET))
                if i < 3:
                    rules.append((f"down_blocks.{i}.attentions.{j}", f"input_blocks.{sgm}.1"))
            if i < 3:
                rules.append((f"down_blocks.{i}.downsamplers.0.conv", f"input_blocks.{3 * i + 3}.0.op"))

        # mid block: resnets.{j} -> middle_block.{2*j}, single attention -> middle_block.1.
        rules.append(("mid_block.resnets.0", "middle_block.0", _RESNET))
        rules.append(("mid_block.attentions.0", "middle_block.1"))
        rules.append(("mid_block.resnets.1", "middle_block.2", _RESNET))

        # up blocks: output_blocks index = 3*i + j; attentions in blocks 1,2,3; upsamplers in 0,1,2. Block 0
        # has no attention so its upsampler is module .1 (after the single resnet); blocks 1,2 have attention
        # so their upsampler is module .2.
        for i in range(4):
            for j in range(3):
                sgm = 3 * i + j
                rules.append((f"up_blocks.{i}.resnets.{j}", f"output_blocks.{sgm}.0", _RESNET))
                if i > 0:
                    rules.append((f"up_blocks.{i}.attentions.{j}", f"output_blocks.{sgm}.1"))
            if i < 3:
                sub = 1 if i == 0 else 2
                rules.append((f"up_blocks.{i}.upsamplers.0.conv", f"output_blocks.{3 * i + 2}.{sub}.conv"))

        return rules

    def lora_diffusers_to_kohya(self) -> list | None:
        # SD's real kohya-ss file keeps diffusers UNet names (sd-scripts only emits sgm for SDXL), so KOHYA
        # uses no rename -- only SD's ORIGINAL/COMFY use the sgm body above.
        return None

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        # Single CLIP under kohya lora_te (no digit). Comfy names it by architecture: SD1.x = clip_l (CLIP-L),
        # SD2.x = clip_h (OpenCLIP-H).
        clip = "clip_l" if self.model_type.is_sd_v1() else "clip_h"
        return [
            (self.text_encoder, {
                ModelFormat.DIFFUSERS_LORA: "text_encoder",
                ModelFormat.KOHYA_LORA: "lora_te",
                ModelFormat.COMFY_LORA: f"text_encoders.{clip}.transformer",
            }),
        ]

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

    def create_pipeline(self, use_original_tokenizers: bool = False) -> DiffusionPipeline:
        tokenizer = self.orig_tokenizer if use_original_tokenizers else self.tokenizer
        if self.model_type.has_depth_input():
            return StableDiffusionDepth2ImgPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                depth_estimator=self.depth_estimator,
                feature_extractor=self.image_depth_processor,
            )
        elif self.model_type.has_conditioning_image_input():
            return StableDiffusionInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=tokenizer,
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
                tokenizer=tokenizer,
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
        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output
