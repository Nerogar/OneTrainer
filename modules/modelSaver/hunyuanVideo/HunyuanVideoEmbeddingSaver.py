import os.path
from pathlib import Path

from modules.model.HunyuanVideoModel import HunyuanVideoModel, HunyuanVideoModelEmbedding
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename

import torch
from torch import Tensor

from safetensors.torch import save_file


class HunyuanVideoEmbeddingSaver:

    def __to_state_dict(
            self,
            embedding: HunyuanVideoModelEmbedding | None,
            embedding_state_dict: dict[str, Tensor] | None,
            dtype: torch.dtype | None,
    ):
        if embedding is None:
            text_encoder_1_vector_cpu = embedding_state_dict.get("llama", None)
            text_encoder_2_vector_cpu = embedding_state_dict.get("clip_l", None)

            text_encoder_1_output_vector_cpu = embedding_state_dict.get("llama_out", None)
            text_encoder_2_output_vector_cpu = embedding_state_dict.get("clip_l_out", None)
        else:
            text_encoder_1_vector_cpu = embedding.text_encoder_1_embedding.vector.to(device="cpu", dtype=dtype) \
                if embedding.text_encoder_1_embedding.vector is not None else None
            text_encoder_2_vector_cpu = embedding.text_encoder_2_embedding.vector.to(device="cpu", dtype=dtype) \
                if embedding.text_encoder_2_embedding.vector is not None else None
            text_encoder_2_output_vector_cpu = embedding.text_encoder_2_embedding.output_vector.to(device="cpu", dtype=dtype) \
                if embedding.text_encoder_2_embedding.output_vector is not None else None
            text_encoder_1_output_vector_cpu = embedding.text_encoder_1_embedding.output_vector.to(device="cpu", dtype=dtype) \
                if embedding.text_encoder_1_embedding.output_vector is not None else None

        state_dict = {}
        if text_encoder_1_vector_cpu is not None:
            state_dict["llama"] = text_encoder_1_vector_cpu
        if text_encoder_2_vector_cpu is not None:
            state_dict["clip_l"] = text_encoder_2_vector_cpu
        if text_encoder_1_output_vector_cpu is not None:
            state_dict["llama_out"] = text_encoder_1_output_vector_cpu
        if text_encoder_2_output_vector_cpu is not None:
            state_dict["clip_l_out"] = text_encoder_2_output_vector_cpu

        return state_dict


    def __save_ckpt(
            self,
            embedding: HunyuanVideoModelEmbedding | None,
            embedding_state_dict: dict[str, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        state_dict = self.__to_state_dict(
            embedding,
            embedding_state_dict,
            dtype,
        )

        torch.save(state_dict, destination)

    def __save_safetensors(
            self,
            embedding: HunyuanVideoModelEmbedding | None,
            embedding_state_dict: dict[str, Tensor] | None,
            destination: str,
            dtype: torch.dtype | None,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        state_dict = self.__to_state_dict(
            embedding,
            embedding_state_dict,
            dtype,
        )

        save_file(state_dict, destination)

    def __save_internal(
            self,
            embedding: HunyuanVideoModelEmbedding | None,
            embedding_state: dict[str, Tensor] | None,
            destination: str,
    ):
        safetensors_embedding_name = os.path.join(
            destination,
            "additional_embeddings",
            f"{embedding.text_encoder_1_embedding.uuid}.safetensors",
        )
        self.__save_safetensors(
            embedding,
            embedding_state,
            safetensors_embedding_name,
            None,
        )

    def save_single(
            self,
            model: HunyuanVideoModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding = model.embedding
        embedding_state = list(model.embedding_state_dicts.values())[0]

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
                )

    def save_multiple(
            self,
            model: HunyuanVideoModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding_uuids = set(model.embedding_state_dicts.keys() \
                              | {x.text_encoder_1_embedding.uuid for x in model.additional_embeddings})

        embeddings = {x.text_encoder_1_embedding.uuid: x for x in model.additional_embeddings}

        for embedding_uuid in embedding_uuids:
            embedding = embeddings.get(embedding_uuid)
            embedding_state = model.embedding_state_dicts.get(embedding_uuid, None)
            embedding_name = safe_filename(embedding.text_encoder_1_embedding.placeholder,
                                           allow_spaces=False, max_length=None)

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
                    )
