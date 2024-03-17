import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.ModelNames import ModelNames, EmbeddingName


class StableDiffusionEmbeddingLoader:
    def __init__(self):
        super(StableDiffusionEmbeddingLoader, self).__init__()

    def __load_embedding(
            self,
            embedding_name: str,
    ) -> Tensor | None:
        if embedding_name == "":
            return None

        try:
            return torch.load(embedding_name)['string_to_param']['*']
        except:
            pass

        try:
            return load_file(embedding_name)["emp_params"]
        except:
            pass

        raise Exception(f"could not load embedding: {embedding_name}")

    def __load_internal(
            self,
            model: StableDiffusionModel,
            base_model_name: str,
            embedding_names: list[EmbeddingName],
    ):
        embedding_states = []

        if os.path.exists(os.path.join(base_model_name, "meta.json")):
            for embedding_name in embedding_names:
                safetensors_embedding_name = os.path.join(
                    base_model_name,
                    "embedding",
                    f"{embedding_name.model_name}.safetensors",
                )

                if os.path.exists(safetensors_embedding_name):
                    embedding_states.append(self.__load_embedding(safetensors_embedding_name))
                else:
                    embedding_states.append(self.__load_embedding(embedding_name.model_name))

            model.additional_embedding_states = embedding_states
        else:
            raise Exception("not an internal model")

    def __load_embeddings(
            self,
            model: StableDiffusionModel,
            embedding_names: list[EmbeddingName],
    ):
        embedding_states = []

        for embedding_name in embedding_names:
            embedding_states.append(self.__load_embedding(embedding_name.model_name))

        model.additional_embedding_states = embedding_states

    def load(
            self,
            model: StableDiffusionModel,
            model_names: ModelNames,
    ):
        stacktraces = []

        try:
            self.__load_internal(model, model_names.base_model, model_names.embeddings)
            return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_embeddings(model, model_names.embeddings)
            return model
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load embedding: " + str(model_names.embedding))
