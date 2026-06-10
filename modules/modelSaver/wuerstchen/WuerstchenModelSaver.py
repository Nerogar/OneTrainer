import copy
import os.path
from pathlib import Path

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_stable_cascade_diffusers_to_ckpt import convert_stable_cascade_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class WuerstchenModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline().prior_pipe
        original_device = pipeline.device
        pipeline.to("cpu")
        pipeline_copy = copy.deepcopy(pipeline)
        pipeline.to(original_device)

        pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        pipeline_copy.save_pretrained(destination)

        del pipeline_copy

    def __save_safetensors(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        if model.model_type.is_stable_cascade():
            os.makedirs(Path(destination).absolute(), exist_ok=True)

            unet_state_dict = convert_stable_cascade_diffusers_to_ckpt(
                model.prior_prior.state_dict(),
            )
            unet_save_state_dict = self._convert_state_dict_dtype(unet_state_dict, dtype)
            self._convert_state_dict_to_contiguous(unet_save_state_dict)
            save_file(
                unet_save_state_dict,
                os.path.join(destination, "stage_c.safetensors"),
                self._create_safetensors_header(model, unet_save_state_dict)
            )

            te_state_dict = model.prior_text_encoder.state_dict()
            te_save_state_dict = self._convert_state_dict_dtype(te_state_dict, dtype)
            self._convert_state_dict_to_contiguous(te_save_state_dict)
            save_file(
                te_save_state_dict,
                os.path.join(destination, "text_encoder.safetensors"),
                self._create_safetensors_header(model, te_save_state_dict)
            )
        else:
            raise NotImplementedError

    def __save_internal(
            self,
            model: WuerstchenModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: WuerstchenModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
