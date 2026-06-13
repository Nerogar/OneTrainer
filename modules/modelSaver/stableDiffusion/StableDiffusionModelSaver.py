import copy
import os.path
from pathlib import Path

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_sd_diffusers_to_ckpt import convert_sd_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch

import yaml
from safetensors.torch import save_file


class StableDiffusionModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: StableDiffusionModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")

        if dtype is not None:
            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)
        else:
            save_pipeline = pipeline

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_sd_diffusers_to_ckpt(
            model_type,
            model.vae.state_dict(),
            model.unet.state_dict(),
            model.text_encoder.state_dict(),
            model.noise_scheduler
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

        yaml_name = os.path.splitext(destination)[0] + '.yaml'
        with open(yaml_name, 'w', encoding='utf8') as f:
            yaml.dump(model.sd_config, f, default_flow_style=False, allow_unicode=True)

    def __save_internal(
            self,
            model: StableDiffusionModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, model_type, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
