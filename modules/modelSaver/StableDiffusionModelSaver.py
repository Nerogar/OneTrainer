import copy
import json
import os.path
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.convert.convert_sd_diffusers_to_ckpt import convert_sd_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class StableDiffusionModelSaver(BaseModelSaver):

    @staticmethod
    def __save_diffusers(
            model: StableDiffusionModel,
            destination: str,
            dtype: torch.dtype,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        original_device = pipeline.device
        pipeline.to("cpu")
        pipeline_copy = copy.deepcopy(pipeline)
        pipeline.to(original_device)

        pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        pipeline_copy.save_pretrained(destination)

        del pipeline_copy

    @staticmethod
    def __save_ckpt(
            model: StableDiffusionModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype,
    ):
        state_dict = convert_sd_diffusers_to_ckpt(
            model_type,
            model.vae.state_dict(),
            model.unet.state_dict(),
            model.text_encoder.state_dict(),
            model.noise_scheduler
        )
        save_state_dict = BaseModelSaver._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

        yaml_name = os.path.splitext(destination)[0] + '.yaml'
        with open(yaml_name, 'w', encoding='utf8') as f:
            yaml.dump(model.sd_config, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def __save_safetensors(
            model: StableDiffusionModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype,
    ):
        state_dict = convert_sd_diffusers_to_ckpt(
            model_type,
            model.vae.state_dict(),
            model.unet.state_dict(),
            model.text_encoder.state_dict(),
            model.noise_scheduler
        )
        save_state_dict = BaseModelSaver._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination)

        yaml_name = os.path.splitext(destination)[0] + '.yaml'
        with open(yaml_name, 'w', encoding='utf8') as f:
            yaml.dump(model.sd_config, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def __save_internal(
            model: StableDiffusionModel,
            destination: str,
    ):
        # base model
        StableDiffusionModelSaver.__save_diffusers(model, destination, torch.float32)

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
                self.__save_ckpt(model, model_type, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, model_type, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
