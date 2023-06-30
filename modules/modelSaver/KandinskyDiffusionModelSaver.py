import copy
import json
import os.path
from pathlib import Path

import torch

from modules.model.BaseModel import BaseModel
from modules.model.KandinskyModel import KandinskyModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class KandinskyModelSaver(BaseModelSaver):

    @staticmethod
    def __save_diffusers(
            model: KandinskyModel,
            prior_destination: str,
            diffusion_destination: str,
            dtype: torch.dtype,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        prior_pipeline = model.create_prior_pipeline()
        original_device = prior_pipeline.device
        prior_pipeline.to("cpu")
        prior_pipeline_copy = copy.deepcopy(prior_pipeline)
        prior_pipeline.to(original_device)

        prior_pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(prior_destination).absolute(), exist_ok=True)
        prior_pipeline_copy.save_pretrained(prior_destination)

        del prior_pipeline_copy

        diffusion_pipeline = model.create_diffusion_pipeline()
        original_device = diffusion_pipeline.device
        diffusion_pipeline.to("cpu")
        diffusion_pipeline_copy = copy.deepcopy(diffusion_pipeline)
        diffusion_pipeline.to(original_device)

        diffusion_pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(diffusion_destination).absolute(), exist_ok=True)
        diffusion_pipeline_copy.save_pretrained(diffusion_destination)

        del diffusion_pipeline_copy

    @staticmethod
    def __save_ckpt(
            model: KandinskyModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype,
    ):
        raise NotImplementedError

    @staticmethod
    def __save_safetensors(
            model: KandinskyModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype,
    ):
        raise NotImplementedError

    @staticmethod
    def __save_internal(
            model: KandinskyModel,
            destination: str,
    ):
        if model.prior_text_encoder.dtype != torch.float32 \
                or model.prior_image_encoder.dtype != torch.float32 \
                or model.prior_prior.dtype != torch.float32 \
                or model.text_encoder.dtype != torch.float32 \
                or model.unet.dtype != torch.float32 \
                or model.movq.dtype != torch.float32:
            # The internal model format requires float32 weights.
            # Other formats don't have the required precision for training.
            raise ValueError("Model weights need to be in float32 format. Something has gone wrong!")

        prior_destination = os.path.join(destination, "prior_model")
        diffusion_destination = os.path.join(destination, "diffusion_model")

        # base model
        KandinskyModelSaver.__save_diffusers(model, prior_destination, diffusion_destination, torch.float32)

        # optimizer
        os.makedirs(os.path.join(destination, "optimizer"), exist_ok=True)
        torch.save(model.optimizer.state_dict(), os.path.join(destination, "optimizer", "optimizer.pt"))

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
                prior_destination, diffusion_destination = output_model_destination.split(';')
                self.__save_diffusers(model, prior_destination, diffusion_destination, dtype)
            case ModelFormat.CKPT:
                self.__save_ckpt(model, model_type, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, model_type, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
