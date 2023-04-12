import os.path
from pathlib import Path

import torch

from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.convert.convert_sd_diffusers_to_ckpt import convert_sd_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class FineTuneModelSaver(BaseModelSaver):

    @staticmethod
    def __convert_dtype(state_dict: dict, dtype: torch.dtype) -> dict:
        converted_state_dict = {}

        for (key, value) in state_dict.items():
            if isinstance(value, dict):
                converted_state_dict[key] = FineTuneModelSaver.__convert_dtype(value, dtype)
            else:
                converted_state_dict[key] = value.clone().detach().to(dtype=dtype)

        return converted_state_dict

    @staticmethod
    def __save_stable_diffusion_ckpt(
            model: StableDiffusionModel,
            destination: str,
            dtype: torch.dtype
    ):
        state_dict = convert_sd_diffusers_to_ckpt(model.vae.state_dict(), model.unet.state_dict(), model.text_encoder.state_dict())
        save_state_dict = FineTuneModelSaver.__convert_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        match model_type:
            case ModelType.STABLE_DIFFUSION_15 \
                 | ModelType.STABLE_DIFFUSION_15_INPAINTING \
                 | ModelType.STABLE_DIFFUSION_20_DEPTH \
                 | ModelType.STABLE_DIFFUSION_20 \
                 | ModelType.STABLE_DIFFUSION_20_INPAINTING:
                match output_model_format:
                    case ModelFormat.DIFFUSERS:
                        raise NotImplementedError
                    case ModelFormat.CKPT:
                        self.__save_stable_diffusion_ckpt(model, output_model_destination, dtype)
                    case ModelFormat.SAFETENSORS:
                        raise NotImplementedError
