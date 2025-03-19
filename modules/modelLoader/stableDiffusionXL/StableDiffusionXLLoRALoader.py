import os
import traceback

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.convert.convert_lora_util import convert_to_diffusers
from modules.util.convert.lora.convert_sdxl_lora import convert_sdxl_lora_key_sets
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class StableDiffusionXLLoRALoader:
    def __init__(self):
        super().__init__()

    def __load_safetensors(
            self,
            model: StableDiffusionXLModel,
            lora_name: str,
    ):
        model.lora_state_dict = convert_to_diffusers(load_file(lora_name), convert_sdxl_lora_key_sets())

    def __load_ckpt(
            self,
            model: StableDiffusionXLModel,
            lora_name: str,
    ):
        model.lora_state_dict = convert_to_diffusers(
            torch.load(lora_name, weights_only=True),
            convert_sdxl_lora_key_sets(),
        )

    def __load_internal(
            self,
            model: StableDiffusionXLModel,
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
            model: StableDiffusionXLModel,
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
