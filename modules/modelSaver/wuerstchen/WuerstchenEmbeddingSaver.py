import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename


class WuerstchenEmbeddingSaver:

    def __save_ckpt(
            self,
            embedding: WuerstchenModelEmbedding,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        prior_text_encoder_vector_cpu = embedding.prior_text_encoder_vector.to(device="cpu", dtype=dtype)

        torch.save(
            {
                "clip_g": prior_text_encoder_vector_cpu,
            },
            destination
        )

    def __save_safetensors(
            self,
            embedding: WuerstchenModelEmbedding,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        prior_text_encoder_vector_cpu = embedding.prior_text_encoder_vector.to(device="cpu", dtype=dtype)

        save_file(
            {
                "clip_g": prior_text_encoder_vector_cpu,
            },
            destination
        )

    def __save_internal(
            self,
            embedding: WuerstchenModelEmbedding,
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
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding = model.embedding

        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(
                    embedding,
                    os.path.join(output_model_destination),
                    dtype,
                )
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(
                    embedding,
                    os.path.join(output_model_destination),
                    dtype,
                )
            case ModelFormat.INTERNAL:
                self.__save_internal(embedding, output_model_destination)

    def save_multiple(
            self,
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        for embedding in model.additional_embeddings:
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
