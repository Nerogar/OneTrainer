import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelSaver.mixin.ModelSaverClipEmbeddingMixin import ModelSaverClipEmbeddingMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename


class StableDiffusionEmbeddingSaver(
    ModelSaverClipEmbeddingMixin,
):

    def __save_ckpt(
            self,
            embedding: StableDiffusionModelEmbedding,
            destination: str,
            dtype: torch.dtype | None,
    ):

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        vector_cpu = embedding.text_encoder_vector.to(device="cpu", dtype=dtype)

        torch.save(
            {
                "string_to_token": {"*": 265},
                "string_to_param": {"*": vector_cpu},
                "name": '*',
                "step": 0,
                "sd_checkpoint": "",
                "sd_checkpoint_name": "",
            },
            destination
        )

    def __save_safetensors(
            self,
            embedding: StableDiffusionModelEmbedding,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        vector_cpu = embedding.text_encoder_vector.to(device="cpu", dtype=dtype)

        save_file(
            {"emp_params": vector_cpu},
            destination
        )

    def __save_internal(
            self,
            embedding: StableDiffusionModelEmbedding,
            destination: str,
    ):
        safetensors_embedding_name = os.path.join(
            destination,
            "embeddings",
            f"{embedding.uuid}.safetensors",
        )
        self.__save_safetensors(embedding, safetensors_embedding_name, None)

    def save_single(
            self,
            model: StableDiffusionModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        embedding = model.embeddings[0]

        embedding_name = safe_filename(embedding.placeholder)

        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(
                    embedding,
                    os.path.join(output_model_destination, f"{embedding_name}.pt"),
                    dtype,
                )
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(
                    embedding,
                    os.path.join(output_model_destination, f"{embedding_name}.safetensors"),
                    dtype,
                )
            case ModelFormat.INTERNAL:
                self.__save_internal(embedding, output_model_destination)

    def save_multiple(
            self,
            model: StableDiffusionModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        for embedding in model.embeddings:
            embedding_name = safe_filename(embedding.placeholder, allow_spaces=False, max_length=None)

            match output_model_format:
                case ModelFormat.DIFFUSERS:
                    raise NotImplementedError
                case ModelFormat.CKPT:
                    self.__save_ckpt(
                        embedding,
                        os.path.join(f"{output_model_destination}_embeddings", f"{embedding_name}.pt"),
                        dtype,
                    )
                case ModelFormat.SAFETENSORS:
                    self.__save_safetensors(
                        embedding,
                        os.path.join(f"{output_model_destination}_embeddings", f"{embedding_name}.safetensors"),
                        dtype,
                    )
                case ModelFormat.INTERNAL:
                    self.__save_internal(embedding, output_model_destination)
