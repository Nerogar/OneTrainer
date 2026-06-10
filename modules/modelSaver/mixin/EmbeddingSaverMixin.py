import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from safetensors.torch import save_file


class EmbeddingSaverMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _to_state_dict(
            self,
            embedding: Any | None,
            embedding_state_dict: dict[str, Tensor] | None,
            dtype: torch.dtype | None,
    ):
        pass

    def _save_safetensors(
            self,
            embedding: Any | None,
            embedding_state_dict: dict[str, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        state_dict = self._to_state_dict(
            embedding,
            embedding_state_dict,
            dtype,
        )

        save_file(state_dict, destination)

    def _save_internal(
            self,
            embedding: Any | None,
            embedding_state: dict[str, Tensor] | None,
            embedding_uuid: str,
            destination: str,
    ):
        safetensors_embedding_name = os.path.join(
            destination,
            "embeddings",
            f"{embedding_uuid}.safetensors",
        )
        self._save_safetensors(
            embedding,
            embedding_state,
            safetensors_embedding_name,
            None,
        )
