from abc import ABCMeta, abstractmethod
from uuid import uuid4

import torch
from torch.optim import Optimizer

from modules.module.EMAModule import EMAModuleWrapper
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class BaseModelEmbedding:
    def __init__(
            self,
            uuid: str,
            token_count: int,
            placeholder: str,
    ):
        self.uuid = uuid
        self.placeholder = placeholder
        self.text_tokens = [f"<{uuid4()}>" for _ in range(token_count)]


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
            embedding: BaseModelEmbedding | None,
            prompt: str,
    ) -> str:
        for embedding in additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        if embedding is not None:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        return prompt
