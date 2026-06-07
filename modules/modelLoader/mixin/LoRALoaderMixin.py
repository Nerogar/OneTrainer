import os
import traceback
from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_diffusers
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class LoRALoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        pass

    @staticmethod
    def scale_lora_state_dict(
            state_dict: dict,
            te_scale: float = 1.0,
            unet_scale: float = 1.0,
    ) -> dict:
        """
        Scales LoRA weights for Text Encoder and main component (UNet/Transformer) separately.
        
        Args:
            state_dict: The LoRA state dict to scale
            te_scale: Scale factor for Text Encoder LoRA weights (default 1.0, applies to lora_te*)
            unet_scale: Scale factor for main component LoRA weights (default 1.0, applies to everything else)
            
        Returns:
            The scaled state dict
        """
        scaled_dict = {}
        
        weight_suffixes = (
            ".weight",
            "hada_w1_a",
            "hada_w1_b",
            "hada_w2_a",
            "hada_w2_b",
            "lokr_w1",
            "lokr_w2",
            "lokr_t1",
            "lokr_t2",
        )

        for key, value in state_dict.items():
            is_weight = isinstance(value, torch.Tensor) and key.endswith(weight_suffixes)
            if key.startswith("lora_te"):
                # Text Encoder LoRA (matches lora_te, lora_te1, lora_te2, etc.)
                scaled_dict[key] = value * te_scale if is_weight else value
            else:
                # Other components: unet, transformer, prior, decoder, etc.
                scaled_dict[key] = value * unet_scale if is_weight else value
        
        return scaled_dict

    def __load_safetensors(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = load_file(lora_name)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            state_dict = convert_to_diffusers(state_dict, key_sets)

        model.lora_state_dict = state_dict

    def __load_ckpt(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = torch.load(lora_name, weights_only=True)

        key_sets = self._get_convert_key_sets(model)
        if key_sets is not None:
            state_dict = convert_to_diffusers(state_dict, key_sets)

        model.lora_state_dict = state_dict

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

        if model_names.lora.endswith(".ckpt"):
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
