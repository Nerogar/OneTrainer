import os
import traceback
from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.convert.convert_lora_util import LoraConversionKeySet, convert_to_diffusers
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class LoRALoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        pass

    def __load_safetensors(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        key_sets = self._get_convert_key_sets(model)

        if key_sets is not None:
            model.lora_state_dict = convert_to_diffusers(load_file(lora_name), key_sets)
        else:
            model.lora_state_dict = load_file(lora_name)

    def __load_ckpt(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        key_sets = self._get_convert_key_sets(model)

        if key_sets is not None:
            model.lora_state_dict = convert_to_diffusers(torch.load(lora_name, weights_only=True), key_sets)
        else:
            model.lora_state_dict = torch.load(lora_name, weights_only=True)

    def __load_internal(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        if os.path.exists(os.path.join(lora_name, "meta.json")):
            safetensors_lora_name = os.path.join(lora_name, "lora", "lora.safetensors")
            if os.path.exists(safetensors_lora_name):
                self.__load_safetensors(model, safetensors_lora_name)
        else:
            raise Exception("not an internal model")

    def _load(
            self,
            model: BaseModel,
            model_names: ModelNames,
    ):
        stacktraces = []

        if model_names.lora == "":
            return

        try:
            self.__load_internal(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_ckpt(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
