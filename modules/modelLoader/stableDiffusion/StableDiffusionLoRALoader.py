import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.ModelLoaderLoRAMixin import ModelLoaderLoRAMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionLoRALoader(
    ModelLoaderLoRAMixin,
):
    def __init__(self):
        super(StableDiffusionLoRALoader, self).__init__()

    def __init_lora(
            self,
            model: StableDiffusionModel,
            state_dict: dict[str, Tensor],
            dtype: torch.dtype,
    ):
        rank = self._get_lora_rank(state_dict)

        model.text_encoder_lora = self._load_lora_with_prefix(
            module=model.text_encoder,
            state_dict=state_dict,
            prefix="lora_te",
            rank=rank,
        )

        model.unet_lora = self._load_lora_with_prefix(
            module=model.unet,
            state_dict=state_dict,
            prefix="lora_unet",
            rank=rank,
            module_filter=["attentions"],
        )

    def __load_safetensors(
            self,
            model: StableDiffusionModel,
            lora_name: str,
    ):
        model.lora_state_dict = load_file(lora_name)

    def __load_ckpt(
            self,
            model: StableDiffusionModel,
            lora_name: str,
    ):
        model.lora_state_dict = torch.load(lora_name)

    def __load_internal(
            self,
            model: StableDiffusionModel,
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
            model: StableDiffusionModel,
            model_names: ModelNames,
    ) -> StableDiffusionModel | None:
        stacktraces = []

        if model_names.lora == "":
            return
        try:
            self.__load_internal(model, model_names.lora)
            return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_ckpt(model, model_names.lora)
            return model
        except:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_names.lora)
            return model
        except:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
