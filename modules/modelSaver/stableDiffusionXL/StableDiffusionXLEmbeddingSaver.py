import os.path
from pathlib import Path

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename

import torch
from torch import Tensor

from safetensors.torch import save_file


class StableDiffusionXLEmbeddingSaver:

    def __save_ckpt(
            self,
            embedding: StableDiffusionXLModelEmbedding | None,
            embedding_state: tuple[Tensor, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if embedding is None:
            text_encoder_1_vector_cpu = embedding_state[0]
            text_encoder_2_vector_cpu = embedding_state[1]
        else:
            text_encoder_1_vector_cpu = embedding.text_encoder_1_vector.to(device="cpu", dtype=dtype)
            text_encoder_2_vector_cpu = embedding.text_encoder_2_vector.to(device="cpu", dtype=dtype)

        torch.save(
            {
                "clip_l": text_encoder_1_vector_cpu,
                "clip_g": text_encoder_2_vector_cpu,
            },
            destination
        )

    def __save_safetensors(
            self,
            embedding: StableDiffusionXLModelEmbedding | None,
            embedding_state: tuple[Tensor, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if embedding is None:
            text_encoder_1_vector_cpu = embedding_state[0]
            text_encoder_2_vector_cpu = embedding_state[1]
        else:
            text_encoder_1_vector_cpu = embedding.text_encoder_1_vector.to(device="cpu", dtype=dtype)
            text_encoder_2_vector_cpu = embedding.text_encoder_2_vector.to(device="cpu", dtype=dtype)

        save_file(
            {
                "clip_l": text_encoder_1_vector_cpu,
                "clip_g": text_encoder_2_vector_cpu,
            },
            destination
        )

    def __save_internal(
            self,
            embedding: StableDiffusionXLModelEmbedding | None,
            embedding_state: tuple[Tensor, Tensor] | None,
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
            model: StableDiffusionXLModel,
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
            model: StableDiffusionXLModel,
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
