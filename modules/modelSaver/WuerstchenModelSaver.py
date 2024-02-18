import copy
import json
import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.BaseModel import BaseModel
from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.convert.convert_stable_cascade_diffusers_to_ckpt import convert_stable_cascade_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class WuerstchenModelSaver(BaseModelSaver):

    def __save_diffusers(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
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

    def __save_internal(
            self,
            model: WuerstchenModel,
            destination: str,
    ):
        # base model
        self.__save_diffusers(model, destination, torch.float32)

        # optimizer
        os.makedirs(os.path.join(destination, "optimizer"), exist_ok=True)
        torch.save(model.optimizer.state_dict(), os.path.join(destination, "optimizer", "optimizer.pt"))

        # ema
        if model.ema:
            os.makedirs(os.path.join(destination, "ema"), exist_ok=True)
            torch.save(model.ema.state_dict(), os.path.join(destination, "ema", "ema.pt"))

        # meta
        with open(os.path.join(destination, "meta.json"), "w") as meta_file:
            json.dump({
                'train_progress': {
                    'epoch': model.train_progress.epoch,
                    'epoch_step': model.train_progress.epoch_step,
                    'epoch_sample': model.train_progress.epoch_sample,
                    'global_step': model.train_progress.global_step,
                },
            }, meta_file)

        # model spec
        with open(os.path.join(destination, "model_spec.json"), "w") as model_spec_file:
            json.dump(self._create_safetensors_header(model), model_spec_file)

    def __save_safetensors(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
    ):
        if model.model_type.is_stable_cascade():
            state_dict = convert_stable_cascade_diffusers_to_ckpt(
                model.prior_prior.state_dict(),
            )
            save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
            self._convert_state_dict_to_contiguous(save_state_dict)

            os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

            save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))
        else:
            raise NotImplementedError

    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.CKPT:
                raise NotImplementedError
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
