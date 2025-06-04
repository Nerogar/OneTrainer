import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from omi_model_standards.convert.lora.convert_lora_util import (
    LoraConversionKeySet,
    convert_to_legacy_diffusers,
    convert_to_omi,
)
from safetensors.torch import save_file


class LoRASaverMixin(
    DtypeModelSaverMixin,
    metaclass=ABCMeta,
):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        pass

    @abstractmethod
    def _get_state_dict(
            self,
            model: BaseModel,
    ) -> dict[str, Tensor]:
        pass

    def __save_safetensors(
            self,
            model: BaseModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            save_state_dict = convert_to_omi(save_state_dict, key_sets)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_legacy_safetensors(
            self,
            model: BaseModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            save_state_dict = convert_to_legacy_diffusers(save_state_dict, key_sets)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: BaseModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        self.__save_safetensors(model, os.path.join(destination, "lora", "lora.safetensors"), None)

    def _save(
            self,
            model: BaseModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
            enable_omi_format: bool = False,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.SAFETENSORS:
                # TODO: remove the enable_omi_format switch and always enable self.__save_safetensors
                if enable_omi_format:
                    self.__save_safetensors(model, output_model_destination, dtype)
                else:
                    self.__save_legacy_safetensors(model, output_model_destination, dtype)
            case ModelFormat.LEGACY_SAFETENSORS:
                self.__save_legacy_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
