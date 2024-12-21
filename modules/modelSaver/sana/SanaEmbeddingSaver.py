import os.path
from pathlib import Path

from modules.model.SanaModel import SanaModel, SanaModelEmbedding
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename

import torch
from torch import Tensor

from safetensors.torch import save_file


class SanaEmbeddingSaver:

    def __save_ckpt(
            self,
            embedding: SanaModelEmbedding | None,
            embedding_state: Tensor | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if embedding is None:
            prior_text_encoder_vector_cpu = embedding_state
        else:
            prior_text_encoder_vector_cpu = embedding.text_encoder_vector.to(device="cpu", dtype=dtype)

        torch.save(
            {
                "gemma": prior_text_encoder_vector_cpu,
            },
            destination
        )

    def __save_safetensors(
            self,
            embedding: SanaModelEmbedding | None,
            embedding_state: Tensor | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if embedding is None:
            prior_text_encoder_vector_cpu = embedding_state
        else:
            prior_text_encoder_vector_cpu = embedding.text_encoder_vector.to(device="cpu", dtype=dtype)

        save_file(
            {
                "gemma": prior_text_encoder_vector_cpu,
            },
            destination
        )

    def __save_internal(
            self,
            embedding: SanaModelEmbedding | None,
            embedding_state: Tensor | None,
            destination: str,
            save_single: bool,
    ):
        if save_single:
            safetensors_embedding_name = os.path.join(
                destination,
                "embedding",
                "embedding.safetensors",
            )
        else:
            safetensors_embedding_name = os.path.join(
                destination,
                "additional_embeddings",
                f"{embedding.uuid}.safetensors",
            )
        self.__save_safetensors(
            embedding,
            embedding_state,
            safetensors_embedding_name,
            None,
        )

    def save_single(
            self,
            model: SanaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding = model.embedding
        embedding_state = model.embedding_state

        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(
                    embedding,
                    embedding_state,
                    os.path.join(output_model_destination),
                    dtype,
                )
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(
                    embedding,
                    embedding_state,
                    os.path.join(output_model_destination),
                    dtype,
                )
            case ModelFormat.INTERNAL:
                self.__save_internal(
                    embedding,
                    embedding_state,
                    output_model_destination,
                    True,
                )

    def save_multiple(
            self,
            model: SanaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        for i in range(max(len(model.additional_embeddings), len(model.additional_embedding_states))):
            embedding = model.additional_embeddings[i] if i < len(model.additional_embeddings) else None
            embedding_state = \
                model.additional_embedding_states[i] if i < len(model.additional_embedding_states) else None
            embedding_name = safe_filename(embedding.placeholder, allow_spaces=False, max_length=None)

            match output_model_format:
                case ModelFormat.DIFFUSERS:
                    raise NotImplementedError
                case ModelFormat.CKPT:
                    self.__save_ckpt(
                        embedding,
                        embedding_state,
                        os.path.join(f"{output_model_destination}_embeddings", f"{embedding_name}.pt"),
                        dtype,
                    )
                case ModelFormat.SAFETENSORS:
                    self.__save_safetensors(
                        embedding,
                        embedding_state,
                        os.path.join(f"{output_model_destination}_embeddings", f"{embedding_name}.safetensors"),
                        dtype,
                    )
                case ModelFormat.INTERNAL:
                    self.__save_internal(
                        embedding,
                        embedding_state,
                        output_model_destination,
                        False,
                    )
