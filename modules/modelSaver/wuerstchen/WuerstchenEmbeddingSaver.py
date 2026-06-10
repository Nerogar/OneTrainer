import os.path
from copy import copy

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.path_util import safe_filename

import torch
from torch import Tensor


class WuerstchenEmbeddingSaver(
    EmbeddingSaverMixin
):
    def __init__(self):
        super().__init__()

    def _to_state_dict(
            self,
            embedding: WuerstchenModelEmbedding | None,
            embedding_state_dict: dict[str, Tensor] | None,
            dtype: torch.dtype | None,
    ):
        state_dict = copy(embedding_state_dict) if embedding_state_dict is not None else {}

        if embedding is not None:
            if embedding.prior_text_encoder_embedding.vector is not None:
                state_dict["clip_g"] = embedding.prior_text_encoder_embedding.vector.to(device="cpu", dtype=dtype)
            if embedding.prior_text_encoder_embedding.output_vector is not None:
                state_dict["clip_g_out"] = embedding.prior_text_encoder_embedding.output_vector.to(device="cpu", dtype=dtype)

        return state_dict

    def save_single(
            self,
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding_uuid = list(model.embedding_state_dicts.keys())[0] if model.embedding is None \
            else model.embedding.prior_text_encoder_embedding.uuid

        embedding = model.embedding
        embedding_state = list(model.embedding_state_dicts.values())[0]

        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.SAFETENSORS:
                self._save_safetensors(
                    embedding,
                    embedding_state,
                    output_model_destination,
                    dtype,
                )
            case ModelFormat.INTERNAL:
                self._save_internal(
                    embedding,
                    embedding_state,
                    embedding_uuid,
                    output_model_destination,
                )

    def save_multiple(
            self,
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding_uuids = set(model.embedding_state_dicts.keys() \
                              | {x.prior_text_encoder_embedding.uuid for x in model.additional_embeddings})

        if model.embedding is not None:
            embedding_uuids.discard(model.embedding.prior_text_encoder_embedding.uuid)

        embeddings = {x.prior_text_encoder_embedding.uuid: x for x in model.additional_embeddings}

        for embedding_uuid in embedding_uuids:
            embedding = embeddings.get(embedding_uuid)
            embedding_state = model.embedding_state_dicts.get(embedding_uuid, None)

            if embedding is None and embedding_state is None:
                continue

            embedding_name = safe_filename(embedding.prior_text_encoder_embedding.placeholder,
                                           allow_spaces=False, max_length=None)

            match output_model_format:
                case ModelFormat.DIFFUSERS:
                    raise NotImplementedError
                case ModelFormat.SAFETENSORS:
                    self._save_safetensors(
                        embedding,
                        embedding_state,
                        os.path.join(f"{output_model_destination}_embeddings", f"{embedding_name}.safetensors"),
                        dtype,
                    )
                case ModelFormat.INTERNAL:
                    self._save_internal(
                        embedding,
                        embedding_state,
                        embedding_uuid,
                        output_model_destination,
                    )
