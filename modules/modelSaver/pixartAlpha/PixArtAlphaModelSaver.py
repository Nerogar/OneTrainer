import os.path
from pathlib import Path

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_pixart_diffusers_to_ckpt import convert_pixart_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class PixArtAlphaModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: PixArtAlphaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline(use_original_tokenizers=True)
        pipeline.to("cpu")

        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.tokenizer)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: PixArtAlphaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_pixart_diffusers_to_ckpt(
            model.model_type,
            model.transformer.state_dict(),
            model.checkpoint_diffusers_to_original(),
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
            case ModelFormat.LEGACY_SAFETENSORS | ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
