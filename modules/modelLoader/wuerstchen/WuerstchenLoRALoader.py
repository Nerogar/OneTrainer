import os
import traceback

from modules.model.WuerstchenModel import WuerstchenModel
from modules.util.convert.convert_stable_cascade_lora_ckpt_to_diffusers import (
    convert_stable_cascade_lora_ckpt_to_diffusers,
)
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class WuerstchenLoRALoader:
    def __init__(self):
        super(WuerstchenLoRALoader, self).__init__()

    def __load_safetensors(
            self,
            model: WuerstchenModel,
            lora_name: str,
    ):
        model.lora_state_dict = load_file(lora_name)
        if model.model_type.is_stable_cascade():
            model.lora_state_dict = convert_stable_cascade_lora_ckpt_to_diffusers(model.lora_state_dict)

    def __load_ckpt(
            self,
            model: WuerstchenModel,
            lora_name: str,
    ):
        model.lora_state_dict = torch.load(lora_name)
        if model.model_type.is_stable_cascade():
            model.lora_state_dict = convert_stable_cascade_lora_ckpt_to_diffusers(model.lora_state_dict)

    def __load_internal(
            self,
            model: WuerstchenModel,
            lora_name: str,
    ):
        if os.path.exists(os.path.join(lora_name, "meta.json")):
            safetensors_lora_name = os.path.join(lora_name, "lora", "lora.safetensors")
            if os.path.exists(safetensors_lora_name):
                self.__load_safetensors(model, safetensors_lora_name)
        else:
            raise Exception("not an internal model")

    def load(
            self,
            model: WuerstchenModel,
            model_names: ModelNames,
    ):
        stacktraces = []

        if model_names.lora == "":
            return

        try:
            self.__load_internal(model, model_names.lora)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_ckpt(model, model_names.lora)
            return
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_names.lora)
            return
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
