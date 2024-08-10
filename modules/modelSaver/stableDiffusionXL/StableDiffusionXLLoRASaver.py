import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat


class StableDiffusionXLLoRASaver(
    DtypeModelSaverMixin,
):

    def __get_state_dict(
            self,
            model: StableDiffusionXLModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_1_lora is not None:
            state_dict |= model.text_encoder_1_lora.state_dict()
        if model.text_encoder_2_lora is not None:
            state_dict |= model.text_encoder_2_lora.state_dict()
        if model.unet_lora is not None:
            state_dict |= model.unet_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                state_dict[f"bundle_emb.{embedding.placeholder}.clip_l"] = embedding.text_encoder_1_vector
                state_dict[f"bundle_emb.{embedding.placeholder}.clip_g"] = embedding.text_encoder_2_vector

        return state_dict

    def __save_ckpt(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

    def __save_safetensors(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: StableDiffusionXLModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        self.__save_safetensors(model, os.path.join(destination, "lora", "lora.safetensors"), None)

    def save(
            self,
            model: StableDiffusionXLModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
