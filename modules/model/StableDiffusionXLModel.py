from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip, encode_clip_chunked, get_num_clip_chunks
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import (
    rescale_noise_scheduler_to_zero_terminal_snr,
)
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class StableDiffusionXLModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor | None,
            text_encoder_2_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_1_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_1_vector,
            is_output_embedding=is_output_embedding,
        )

        self.text_encoder_2_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_2_vector,
            is_output_embedding=is_output_embedding,
        )


class StableDiffusionXLModel(BaseModel):
    # base model data
    tokenizer_1: CLIPTokenizer | None
    tokenizer_2: CLIPTokenizer | None
    noise_scheduler: DDIMScheduler | None
    text_encoder_1: CLIPTextModel | None
    text_encoder_2: CLIPTextModelWithProjection | None
    vae: AutoencoderKL | None
    unet: UNet2DConditionModel | None

    # autocast context
    vae_autocast_context: torch.autocast | nullcontext

    vae_train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusionXLModelEmbedding | None
    additional_embeddings: list[StableDiffusionXLModelEmbedding] | None
    embedding_wrapper_1: AdditionalEmbeddingWrapper | None
    embedding_wrapper_2: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
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

        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.noise_scheduler = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.vae = None
        self.unet = None

        self.vae_autocast_context = nullcontext()

        self.vae_train_dtype = DataType.FLOAT_32

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper_1 = None
        self.embedding_wrapper_1 = None

        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None
        self.lora_state_dict = None

        self.sd_config = None
        self.sd_config_filename = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_1_lora,
            self.text_encoder_2_lora,
            self.unet_lora,
        ] if a is not None]

    def all_embeddings(self) -> list[StableDiffusionXLModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_1_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_1_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_1_embedding] if self.embedding is not None else [])

    def all_text_encoder_2_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_2_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_2_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)
        self.text_encoder_2.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def text_encoder_1_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device):
        self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

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

    def force_v_prediction(self):
        self.noise_scheduler.config.prediction_type = 'v_prediction'
        self.sd_config['model']['params']['parameterization'] = 'v'
        self.model_spec.prediction_type = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        self.sd_config['model']['params']['parameterization'] = 'epsilon'
        self.model_spec.prediction_type = 'epsilon'

    def rescale_noise_scheduler_to_zero_terminal_snr(self):
        rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)

    def add_text_encoder_1_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_1_embeddings(), prompt)

    def add_text_encoder_2_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_2_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            pooled_text_encoder_2_output: Tensor = None,
            min_chunks: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        use_chunking = self.train_config.use_clip_token_chunks if self.train_config else False
        encode_fn = encode_clip_chunked if use_chunking else encode_clip

        max_chunks = self.train_config.clip_max_chunks if self.train_config else 1
        chunk_size = self.text_encoder_1.config.max_position_embeddings - 2
        max_length = max_chunks * chunk_size + 2

        if tokens_1 is None and text is not None:
            tokenizer_output = self.tokenizer_1(
                self.add_text_encoder_1_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=max(self.tokenizer_1.model_max_length, max_length) if use_chunking else self.tokenizer_1.model_max_length,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokenizer_output = self.tokenizer_2(
                self.add_text_encoder_2_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=max(self.tokenizer_2.model_max_length, max_length) if use_chunking else self.tokenizer_2.model_max_length,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)

        text_encoder_1_kwargs = {}
        text_encoder_2_kwargs = {}
        if use_chunking:
            chunk_size_1 = self.text_encoder_1.config.max_position_embeddings - 2
            chunk_size_2 = self.text_encoder_2.config.max_position_embeddings - 2

            split_on_comma = self.train_config.clip_chunk_split_on_comma if self.train_config else False
            min_chunks_1 = get_num_clip_chunks(tokens_1, chunk_size_1, split_on_comma, self.tokenizer_1, max_chunks=max_chunks)
            min_chunks_2 = get_num_clip_chunks(tokens_2, chunk_size_2, split_on_comma, self.tokenizer_2, max_chunks=max_chunks)

            min_chunks = max(min_chunks, min_chunks_1, min_chunks_2)

            text_encoder_1_kwargs['min_chunks'] = min_chunks
            text_encoder_1_kwargs['max_chunks'] = max_chunks
            text_encoder_2_kwargs['min_chunks'] = min_chunks
            text_encoder_2_kwargs['max_chunks'] = max_chunks
            text_encoder_1_kwargs['pooled_output_handling'] = self.train_config.clip_chunk_pooled_output_handling
            text_encoder_2_kwargs['pooled_output_handling'] = self.train_config.clip_chunk_pooled_output_handling
            text_encoder_1_kwargs['split_on_comma'] = self.train_config.clip_chunk_split_on_comma
            text_encoder_1_kwargs['tokenizer'] = self.tokenizer_1
            text_encoder_2_kwargs['split_on_comma'] = self.train_config.clip_chunk_split_on_comma
            text_encoder_2_kwargs['tokenizer'] = self.tokenizer_2

        text_encoder_1_output, _ = encode_fn(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-2,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            add_pooled_output=False,
            use_attention_mask=False,
            add_layer_norm=False,
            **text_encoder_1_kwargs,
        )

        text_encoder_2_output, pooled_text_encoder_2_output = encode_fn(
            text_encoder=self.text_encoder_2,
            tokens=tokens_2,
            default_layer=-2,
            layer_skip=text_encoder_2_layer_skip,
            text_encoder_output=text_encoder_2_output,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_2_output,
            use_attention_mask=False,
            add_layer_norm=False,
            **text_encoder_2_kwargs,
        )

        text_encoder_1_output = self._apply_output_embeddings(
            self.all_text_encoder_1_embeddings(),
            self.tokenizer_1,
            tokens_1,
            text_encoder_1_output,
            use_clip_token_chunks=use_chunking,
            clip_max_chunks=max_chunks,
        )

        text_encoder_2_output = self._apply_output_embeddings(
            self.all_text_encoder_2_embeddings(),
            self.tokenizer_2,
            tokens_2,
            text_encoder_2_output,
            use_clip_token_chunks=use_chunking,
            clip_max_chunks=max_chunks,
        )

        # apply dropout
        if text_encoder_1_dropout_probability is not None and text_encoder_1_dropout_probability > 0.0:
            dropout_text_encoder_1_mask = (torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask[:, None, None]

        if text_encoder_2_dropout_probability is not None and text_encoder_2_dropout_probability > 0.0:
            dropout_text_encoder_2_mask = (torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask[:, None]
            text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask[:, None, None]

        return text_encoder_1_output, text_encoder_2_output, pooled_text_encoder_2_output

    def combine_text_encoder_output(
            self,
            text_encoder_1_output: Tensor,
            text_encoder_2_output: Tensor,
            pooled_text_encoder_2_output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)
        return text_encoder_output, pooled_text_encoder_2_output
