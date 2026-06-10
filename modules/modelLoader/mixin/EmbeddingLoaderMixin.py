import contextlib
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.ModelNames import EmbeddingName, ModelNames

import torch
from torch import Tensor

from safetensors.torch import load_file


class EmbeddingLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def __load_embedding(
            self,
            embedding_name: str,
    ) -> dict[str, Tensor] | None:
        if embedding_name == "":
            return None

        with contextlib.suppress(Exception):
            return torch.load(embedding_name, weights_only=True)

        with contextlib.suppress(Exception):
            return load_file(embedding_name)

        raise Exception(f"could not load embedding: {embedding_name}")

    def __load_internal(
            self,
            directory: str,
            embedding_name: EmbeddingName,
    ) -> dict[str, Tensor] | None:
        if os.path.exists(os.path.join(directory, "meta.json")):
            safetensors_embedding_name = os.path.join(
                directory,
                "embeddings",
                f"{embedding_name.uuid}.safetensors",
            )

            if os.path.exists(safetensors_embedding_name):
                return self.__load_embedding(safetensors_embedding_name)
            else:
                return self.__load_embedding(embedding_name.model_name)
        else:
            raise Exception("not an internal model")

    def _load(
            self,
            model: BaseModel,
            directory: str,
            model_names: ModelNames,
    ):
        for embedding_name in model_names.all_embedding():
            try:
                model.embedding_state_dicts[embedding_name.uuid] = \
                    self.__load_internal(directory, embedding_name)
            except Exception as e1:  # noqa: PERF203
                try:
                    model.embedding_state_dicts[embedding_name.uuid] = \
                        self.__load_embedding(embedding_name.model_name)
                except Exception as e2:
                    e2.__cause__ = e1
                    raise Exception(f"could not load embedding: {embedding_name}") from e2
