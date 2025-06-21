import copy
import os.path
from pathlib import Path

from modules.model.FluxModel import FluxModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_flux_diffusers_to_ckpt import convert_flux_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat

import torch

from transformers import T5EncoderModel

from safetensors.torch import save_file


class FluxModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: FluxModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        if dtype is not None:
            # replace the tokenizers __deepcopy__ before calling deepcopy, to prevent a copy being made.
            # the tokenizer tries to reload from the file system otherwise
            tokenizer_2 = pipeline.tokenizer_2
            tokenizer_2.__deepcopy__ = lambda memo: tokenizer_2

            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)

            delattr(tokenizer_2, '__deepcopy__')
        else:
            save_pipeline = pipeline

        text_encoder_2 = save_pipeline.text_encoder_2
        if text_encoder_2 is not None:
            text_encoder_2_save_pretrained = text_encoder_2.save_pretrained
            def save_pretrained_t5(
                    self,
                    *args,
                    **kwargs,
            ):
                # Saving a safetensors file copies all tensors in RAM.
                # Setting the max_shard_size to 2GB reduces this memory overhead a bit.
                # This parameter is set by patching the function, because it's not exposed to the pipeline.
                kwargs = dict(kwargs)
                kwargs['max_shard_size'] = '2GB'
                text_encoder_2_save_pretrained(*args, **kwargs)

            text_encoder_2.save_pretrained = save_pretrained_t5.__get__(text_encoder_2, T5EncoderModel)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if text_encoder_2 is not None:
            text_encoder_2.save_pretrained = text_encoder_2_save_pretrained

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: FluxModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_flux_diffusers_to_ckpt(
            model.transformer.state_dict(),
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: FluxModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: FluxModel,
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
