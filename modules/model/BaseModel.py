from abc import ABCMeta, abstractmethod
from uuid import uuid4

from modules.module.EMAModule import EMAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress

import torch
from torch.optim import Optimizer

from transformers import PreTrainedTokenizer


class BaseModelEmbedding:
    def __init__(
            self,
            uuid: str,
            placeholder: str,
            vector: torch.Tensor | None,
            is_output_embedding: bool,
    ):
        self.uuid = uuid
        self.token_count = vector.shape[0] if vector is not None else 0
        self.placeholder = placeholder
        self.is_output_embedding = is_output_embedding

        if vector is not None:
            if is_output_embedding:
                self.vector = torch.zeros_like(vector).requires_grad_(False)
                self.output_vector = vector.to(dtype=torch.float32)
            else:
                self.vector = vector
                self.output_vector = None
        else:
            self.vector = None
            self.output_vector = None

        self.text_tokens = [f"<{uuid4()}>" for _ in range(self.token_count)]

    def requires_grad(self) -> bool:
        if self.is_output_embedding:
            return self.output_vector.requires_grad
        else:
            return self.vector.requires_grad

    def requires_grad_(self, requires_grad: bool):
        if self.is_output_embedding:
            self.output_vector.requires_grad_(requires_grad)
        else:
            self.vector.requires_grad_(requires_grad)


class BaseModel(metaclass=ABCMeta):
    model_type: ModelType
    parameters: NamedParameterGroupCollection | None
    optimizer: Optimizer | None
    optimizer_state_dict: dict | None
    param_group_mapping: list[str] | None
    ema: EMAModuleWrapper
    ema_state_dict: dict | None
    train_progress: TrainProgress
    model_spec: ModelSpec | None
    train_config: TrainConfig | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        self.model_type = model_type
        self.parameters = None
        self.optimizer = None
        self.optimizer_state_dict = None
        self.param_group_mapping = None
        self.ema_state_dict = None
        self.train_progress = TrainProgress()
        self.model_spec = None
        self.train_config = None

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def eval(self):
        pass

    @staticmethod
    def _add_embeddings_to_prompt(
            additional_embeddings: list[BaseModelEmbedding],
            prompt: str,
    ) -> str:
        for embedding in additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        return prompt


    @staticmethod
    def _replace_tokens(
            tokenizer: PreTrainedTokenizer,
            embedding_text_tokens: str,
            embedding_vectors: torch.Tensor,
            tokens: torch.Tensor,
            text_encoder_output: torch.Tensor,
    ):
        embedding_tokens = tokenizer(
            embedding_text_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(text_encoder_output.device)

        batch_size = text_encoder_output.shape[0]

        embedding_tokens = embedding_tokens.expand(batch_size, -1)
        idx_0, idx_1, idx_2 = (tokens.unsqueeze(1) == embedding_tokens.unsqueeze(2)).nonzero(as_tuple=True)
        text_encoder_output[idx_0, idx_2] = embedding_vectors[idx_1]
