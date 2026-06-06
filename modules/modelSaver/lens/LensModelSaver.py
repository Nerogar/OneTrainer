import copy
import os.path
from pathlib import Path

from modules.model.LensModel import LensModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class LensModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: LensModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # this is the only save path that embeds the base text encoder, so the on-demand encoder is
        # loaded here (and freed again afterwards). resident encoders are untouched.
        model.materialize_text_encoder_for_save()
        try:
            # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
            pipeline = model.create_pipeline()
            pipeline.to("cpu")
            if dtype is not None: #TODO necessary?
                # replace the tokenizers __deepcopy__ before calling deepcopy, to prevent a copy being made.
                # the tokenizer tries to reload from the file system otherwise
                tokenizer = pipeline.tokenizer
                tokenizer.__deepcopy__ = lambda memo: tokenizer

                save_pipeline = copy.deepcopy(pipeline)
                save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)

                delattr(tokenizer, '__deepcopy__')
            else:
                save_pipeline = pipeline

            os.makedirs(Path(destination).absolute(), exist_ok=True)
            save_pipeline.save_pretrained(destination)

            if dtype is not None:
                del save_pipeline
        finally:
            model.release_text_encoder()

    def __save_safetensors(
            self,
            model: LensModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Lens transformer uses diffusers-format keys; no key conversion needed.
        state_dict = model.transformer.state_dict()
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: LensModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: LensModel,
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
