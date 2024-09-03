from contextlib import nullcontext

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class StableDiffusionXLModelEmbedding(BaseModelEmbedding):
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor,
            text_encoder_2_vector: Tensor,
            placeholder: str,
    ):
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder,
        )

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector


class StableDiffusionXLModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    vae: AutoencoderKL
    unet: UNet2DConditionModel

    # autocast context
    autocast_context: torch.autocast | nullcontext
    vae_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    vae_train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusionXLModelEmbedding | None
    embedding_state: tuple[Tensor, Tensor] | None
    additional_embeddings: list[StableDiffusionXLModelEmbedding] | None
    additional_embedding_states: list[tuple[Tensor, Tensor] | None]
    embedding_wrapper_1: AdditionalEmbeddingWrapper
    embedding_wrapper_2: AdditionalEmbeddingWrapper

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
            tokenizer_1: CLIPTokenizer | None = None,
            tokenizer_2: CLIPTokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder_1: CLIPTextModel | None = None,
            text_encoder_2: CLIPTextModelWithProjection | None = None,
            vae: AutoencoderKL | None = None,
            unet: UNet2DConditionModel | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: StableDiffusionXLModelEmbedding | None = None,
            embedding_state: tuple[Tensor, Tensor] | None = None,
            additional_embeddings: list[StableDiffusionXLModelEmbedding] | None = None,
            additional_embedding_states: list[tuple[Tensor, Tensor] | None] = None,
            embedding_wrapper_1: AdditionalEmbeddingWrapper | None = None,
            embedding_wrapper_2: AdditionalEmbeddingWrapper | None = None,
            text_encoder_1_lora: LoRAModuleWrapper | None = None,
            text_encoder_2_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            lora_state_dict: dict | None = None,
            sd_config: dict | None = None,
            sd_config_filename: str | None = None,
            model_spec: ModelSpec | None = None,
            train_config: TrainConfig | None = None,
    ):
        super(StableDiffusionXLModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
            train_config=train_config,
        )

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.unet = unet

        self.autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.vae_train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper_1 = embedding_wrapper_1
        self.embedding_wrapper_1 = embedding_wrapper_2

        self.text_encoder_1_lora = text_encoder_1_lora
        self.text_encoder_2_lora = text_encoder_2_lora
        self.unet_lora = unet_lora
        self.lora_state_dict = lora_state_dict

        self.sd_config = sd_config
        self.sd_config_filename = sd_config_filename

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

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.additional_embeddings, self.embedding, prompt)

    def encode_text(
            self,
            text: str = None,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
    ):
        chunk_length = 75
        max_embeddings_multiples = 3

        def __process_tokens(tokens, tokenizer, text_encoder, layer_skip):
            if tokens is None or tokens.numel() == 0:
                return None, None

            chunks = [tokens[:, i:i + chunk_length] for i in range(0, tokens.shape[1], chunk_length)]
            chunk_embeddings = []
            pooled_outputs = []
            attention_masks = []

            for i, chunk in enumerate(chunks):
                if chunk.numel() == 0:
                    continue

                # Create attention mask (1 for non-masked, 0 for masked)
                attention_mask = torch.ones_like(chunk, dtype=torch.bool)

                # First, add BOS and EOS tokens
                bos_tokens = torch.full((chunk.shape[0], 1), tokenizer.bos_token_id, dtype=chunk.dtype, device=chunk.device)
                eos_tokens = torch.full((chunk.shape[0], 1), tokenizer.eos_token_id, dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([bos_tokens, chunk, eos_tokens], dim=1)
                attention_mask = torch.cat([torch.zeros_like(bos_tokens, dtype=torch.bool) if i > 0 else torch.ones_like(bos_tokens, dtype=torch.bool),
                                            attention_mask, 
                                            torch.zeros_like(eos_tokens, dtype=torch.bool) if i < len(chunks) - 1 else torch.ones_like(eos_tokens, dtype=torch.bool)],
                                            dim=1)

                # Fill with padding
                if chunk.shape[1] < chunk_length + 2:  # +2 is for BOS and EOS
                    padding = torch.full((chunk.shape[0], chunk_length + 2 - chunk.shape[1]), tokenizer.eos_token_id, dtype=chunk.dtype, device=chunk.device)
                    chunk = torch.cat([chunk, padding], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(padding, dtype=torch.bool)], dim=1)
                
                attention_masks.append(attention_mask)
                
                with self.autocast_context:
                    outputs = text_encoder(
                        chunk,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    embedding = outputs.hidden_states[-(2 + layer_skip)]
                    if hasattr(outputs, 'text_embeds'):
                        pooled_outputs.append(outputs.text_embeds)

                chunk_embeddings.append(embedding)

            if not chunk_embeddings:
                return None, None

            if len(chunk_embeddings) > max_embeddings_multiples:
                chunk_embeddings = chunk_embeddings[:max_embeddings_multiples]
                attention_masks = attention_masks[:max_embeddings_multiples]
                if pooled_outputs:
                    pooled_outputs = pooled_outputs[:max_embeddings_multiples]

            combined_embedding = torch.cat(chunk_embeddings, dim=1)
            # combined_attention_mask = torch.cat(attention_masks, dim=1)
            pooled_output = pooled_outputs[0] if pooled_outputs else None

            return combined_embedding, pooled_output

        if tokens_1 is None and text is not None:
            tokens_1 = self.tokenizer_1(
                text,
                padding='max_length',
                truncation=False,
                return_tensors="pt",
            ).input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokens_2 = self.tokenizer_2(
                text,
                padding='max_length',
                truncation=False,
                return_tensors="pt",
            ).input_ids.to(self.text_encoder_2.device)

        if text_encoder_1_output is None:
            text_encoder_1_output, _ = __process_tokens(tokens_1, self.tokenizer_1, self.text_encoder_1, text_encoder_1_layer_skip)

        if text_encoder_2_output is None or pooled_text_encoder_2_output is None:
            text_encoder_2_output, pooled_text_encoder_2_output = __process_tokens(tokens_2, self.tokenizer_2, self.text_encoder_2, text_encoder_2_layer_skip)

        if text_encoder_1_output is None or text_encoder_2_output is None:
            print("Both text encoder outputs are None. Check your input text or tokens.")

        text_encoder_output = torch.cat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output
