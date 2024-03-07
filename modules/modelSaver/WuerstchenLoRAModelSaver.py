import json
import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from modules.model.BaseModel import BaseModel
from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.convert.convert_stable_cascade_lora_diffusers_to_ckpt import \
    convert_stable_cascade_lora_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class WuerstchenLoRAModelSaver(BaseModelSaver):

    def __get_state_dict(
            self,
            model: WuerstchenModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.prior_text_encoder_lora is not None:
            state_dict |= model.prior_text_encoder_lora.state_dict()
        if model.prior_prior_lora is not None:
            state_dict |= model.prior_prior_lora.state_dict()

        return state_dict

    def __save_ckpt(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        if model.model_type.is_stable_cascade():
            save_state_dict = convert_stable_cascade_lora_diffusers_to_ckpt(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

    def __save_safetensors(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        if model.model_type.is_stable_cascade():
            save_state_dict = convert_stable_cascade_lora_diffusers_to_ckpt(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: WuerstchenModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        # lora
        self.__save_safetensors(
            model,
            os.path.join(destination, "lora", "lora.safetensors"),
            torch.float32
        )

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
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
