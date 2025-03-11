import os.path
from pathlib import Path

from modules.model.FluxModel import FluxModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_flux_lora import convert_flux_lora_key_sets
from modules.util.convert.convert_lora_util import convert_to_legacy_diffusers, convert_to_omi
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class FluxLoRASaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __get_state_dict(
            self,
            model: FluxModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_1_lora is not None:
            state_dict |= model.text_encoder_1_lora.state_dict()
        if model.text_encoder_2_lora is not None:
            state_dict |= model.text_encoder_2_lora.state_dict()
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_1_embedding.placeholder

                if embedding.text_encoder_1_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l"] = embedding.text_encoder_1_embedding.vector
                if embedding.text_encoder_2_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.t5"] = embedding.text_encoder_2_embedding.vector
                if embedding.text_encoder_1_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l_out"] = embedding.text_encoder_1_embedding.output_vector
                if embedding.text_encoder_2_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.t5_out"] = embedding.text_encoder_2_embedding.output_vector

        return state_dict

    def __save_safetensors(
            self,
            model: FluxModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        save_state_dict = convert_to_omi(save_state_dict, convert_flux_lora_key_sets())

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_legacy_safetensors(
            self,
            model: FluxModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        save_state_dict = convert_to_legacy_diffusers(save_state_dict, convert_flux_lora_key_sets())

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: FluxModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        self.__save_safetensors(model, os.path.join(destination, "lora", "lora.safetensors"), None)

    def save(
            self,
            model: FluxModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.LEGACY_SAFETENSORS:
                self.__save_legacy_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
