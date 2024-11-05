import contextlib
import os

from modules.model.WuerstchenModel import WuerstchenModel
from modules.util.ModelNames import EmbeddingName, ModelNames

import torch
from torch import Tensor

from safetensors.torch import load_file


class WuerstchenEmbeddingLoader:
    def __init__(self):
        super().__init__()

    def __load_embedding(
            self,
            embedding_name: str,
    ) -> Tensor | None:
        if embedding_name == "":
            return None

        with contextlib.suppress(Exception):
            embedding_state = torch.load(embedding_name, weights_only=True)

            prior_text_encoder_vector = embedding_state['clip_g']

            return prior_text_encoder_vector

        with contextlib.suppress(Exception):
            embedding_state = load_file(embedding_name)

            prior_text_encoder_vector = embedding_state['clip_g']

            return prior_text_encoder_vector

        raise Exception(f"could not load embedding: {embedding_name}")

    def __load_internal(
            self,
            directory: str,
            embedding_name: EmbeddingName,
            load_single: bool,
    ) -> Tensor | None:
        if os.path.exists(os.path.join(directory, "meta.json")):
            if load_single:
                safetensors_embedding_name = os.path.join(
                    directory,
                    "embedding",
                    "embedding.safetensors",
                )
            else:
                safetensors_embedding_name = os.path.join(
                    directory,
                    "embedding",
                    f"{embedding_name.uuid}.safetensors",
                )

            if os.path.exists(safetensors_embedding_name):
                return self.__load_embedding(safetensors_embedding_name)
            else:
                return self.__load_embedding(embedding_name.model_name)
        else:
            raise Exception("not an internal model")

    def load_multiple(
            self,
            model: WuerstchenModel,
            model_names: ModelNames,
    ):
        model.additional_embedding_states = []

        for embedding_name in model_names.additional_embeddings:
            try:
                model.additional_embedding_states.append(self.__load_internal(model_names.base_model, embedding_name, False))
            except Exception as e1:  # noqa: PERF203
                try:
                    model.additional_embedding_states.append(self.__load_embedding(embedding_name.model_name))
                except Exception as e2:
                    e2.__cause__ = e1
                    raise Exception(f"could not load embedding: {embedding_name}") from e2

    def load_single(
            self,
            model: WuerstchenModel,
            model_names: ModelNames,
    ):
        embedding_name = model_names.embedding

        try:
            model.embedding_state = self.__load_internal(model_names.embedding.model_name, embedding_name, True)
        except Exception as e1:
            try:
                model.embedding_state = self.__load_embedding(embedding_name.model_name)
            except Exception as e2:
                e2.__cause__ = e1
                raise Exception(f"could not load embedding: {embedding_name}") from e2
