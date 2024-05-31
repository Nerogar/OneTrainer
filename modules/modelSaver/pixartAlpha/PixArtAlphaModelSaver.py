import copy
import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_pixart_diffusers_to_ckpt import convert_pixart_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat


class PixArtAlphaModelSaver(
    DtypeModelSaverMixin,
):

    def __save_diffusers(
            self,
            model: PixArtAlphaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        original_device = pipeline.device
        pipeline.to("cpu")

        # replace the tokenizers __deepcopy__ before calling deepcopy, to prevent a copy being made.
        # the tokenizer tries to reload from the file system otherwise
        tokenizer = pipeline.tokenizer
        tokenizer.__deepcopy__ = lambda memo: tokenizer
        pipeline_copy = copy.deepcopy(pipeline)
        delattr(tokenizer, '__deepcopy__')
        pipeline.to(original_device)

        pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        pipeline_copy.save_pretrained(destination)

        del pipeline_copy

    def __save_ckpt(
            self,
            model: PixArtAlphaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_pixart_diffusers_to_ckpt(
            model.model_type,
            model.transformer.state_dict(),
        )

        state_dict = {'state_dict': state_dict}

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

    def __save_safetensors(
            self,
            model: PixArtAlphaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_pixart_diffusers_to_ckpt(
            model.model_type,
            model.transformer.state_dict(),
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: PixArtAlphaModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: PixArtAlphaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.CKPT:
                self.__save_ckpt(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
