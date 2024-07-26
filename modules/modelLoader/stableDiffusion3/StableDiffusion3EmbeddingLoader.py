import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.util.ModelNames import ModelNames, EmbeddingName


class StableDiffusion3EmbeddingLoader:
    def __init__(self):
        super(StableDiffusion3EmbeddingLoader, self).__init__()

    def __load_embedding(
            self,
            embedding_name: str,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        if embedding_name == "":
            return None

        try:
            embedding_state = torch.load(embedding_name)

            text_encoder_1_vector = embedding_state['clip_l'] if 'clip_l' in embedding_state else None
            text_encoder_2_vector = embedding_state['clip_g'] if 'clip_g' in embedding_state else None
            text_encoder_3_vector = embedding_state['t5'] if 't5' in embedding_state else None

            return text_encoder_1_vector, text_encoder_2_vector, text_encoder_3_vector
        except:
            pass

        try:
            embedding_state = load_file(embedding_name)

            text_encoder_1_vector = embedding_state['clip_l'] if 'clip_l' in embedding_state else None
            text_encoder_2_vector = embedding_state['clip_g'] if 'clip_g' in embedding_state else None
            text_encoder_3_vector = embedding_state['t5'] if 't5' in embedding_state else None

            return text_encoder_1_vector, text_encoder_2_vector, text_encoder_3_vector
        except:
            pass

        raise Exception(f"could not load embedding: {embedding_name}")

    def __load_internal(
            self,
            directory: str,
            embedding_name: EmbeddingName,
            load_single: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if os.path.exists(os.path.join(directory, "meta.json")):
            if load_single:
                safetensors_embedding_name = os.path.join(
                    directory,
                    "embedding",
                    f"embedding.safetensors",
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
            model: StableDiffusion3Model,
            model_names: ModelNames,
    ):
        model.additional_embedding_states = []

        for embedding_name in model_names.additional_embeddings:
            stacktraces = []

            try:
                model.additional_embedding_states.append(self.__load_internal(model_names.base_model, embedding_name, False))
                continue
            except:
                try:
                    model.additional_embedding_states.append(self.__load_embedding(embedding_name.model_name))
                    continue
                except:
                    stacktraces.append(traceback.format_exc())

                stacktraces.append(traceback.format_exc())

                for stacktrace in stacktraces:
                    print(stacktrace)
                raise Exception("could not load embedding: " + str(model_names.embedding))

    def load_single(
            self,
            model: StableDiffusion3Model,
            model_names: ModelNames,
    ):
        stacktraces = []

        embedding_name = model_names.embedding

        try:
            model.embedding_state = self.__load_internal(model_names.embedding.model_name, embedding_name, True)
            return
        except:
            stacktraces.append(traceback.format_exc())

            try:
                model.embedding_state = self.__load_embedding(embedding_name.model_name)
                return
            except:
                stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load embedding: " + str(model_names.embedding))
