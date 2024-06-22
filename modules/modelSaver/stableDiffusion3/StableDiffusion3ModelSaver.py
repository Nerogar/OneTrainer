import copy
import os.path
from pathlib import Path

import torch
from transformers import T5EncoderModel

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat


class StableDiffusion3ModelSaver(
    DtypeModelSaverMixin,
):

    def __save_diffusers(
            self,
            model: StableDiffusion3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        if dtype is not None:
            # replace the tokenizers __deepcopy__ before calling deepcopy, to prevent a copy being made.
            # the tokenizer tries to reload from the file system otherwise
            tokenizer_3 = pipeline.tokenizer_3
            tokenizer_3.__deepcopy__ = lambda memo: tokenizer_3

            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)

            delattr(tokenizer_3, '__deepcopy__')
        else:
            save_pipeline = pipeline

        text_encoder_3 = save_pipeline.text_encoder_3
        if text_encoder_3 is not None:
            text_encoder_3_save_pretrained = text_encoder_3.save_pretrained
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
                text_encoder_3_save_pretrained(*args, **kwargs)

            text_encoder_3.save_pretrained = save_pretrained_t5.__get__(text_encoder_3, T5EncoderModel)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if text_encoder_3 is not None:
            text_encoder_3.save_pretrained = text_encoder_3_save_pretrained

        if dtype is not None:
            del save_pipeline

    def __save_ckpt(
            self,
            model: StableDiffusion3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        pass

    def __save_safetensors(
            self,
            model: StableDiffusion3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        pass

    def __save_internal(
            self,
            model: StableDiffusion3Model,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: StableDiffusion3Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
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
