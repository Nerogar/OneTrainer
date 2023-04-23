import json
import os.path
from pathlib import Path

import torch

from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class StableDiffusionEmbeddingModelSaver(BaseModelSaver):

    @staticmethod
    def __save_ckpt(
            model: StableDiffusionModel,
            destination: str,
            dtype: torch.dtype,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        vector_cpu = model.embeddings[0].vector.to("cpu", dtype)

        torch.save(
            {
                "string_to_token": {"*": 265},
                "string_to_param": {"*": vector_cpu},
                "name": model.embeddings[0].name,
                "step": 0,
                "sd_checkpoint": "",
                "sd_checkpoint_name": "",
            },
            destination
        )

    @staticmethod
    def __save_internal(
            model: StableDiffusionModel,
            destination: str,
    ):
        if model.embeddings[0].vector.dtype != torch.float32:
            # The internal model format requires float32 weights.
            # Other formats don't have the required precision for training.
            raise ValueError("Model weights need to be in float32 format. Something has gone wrong!")

        os.makedirs(destination, exist_ok=True)

        # embedding
        StableDiffusionEmbeddingModelSaver.__save_ckpt(
            model,
            os.path.join(destination, "embedding", "embedding.pt"),
            torch.float32
        )

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
        if model_type.is_stable_diffusion():
            match output_model_format:
                case ModelFormat.DIFFUSERS:
                    raise NotImplementedError
                case ModelFormat.CKPT:
                    self.__save_ckpt(model, output_model_destination, dtype)
                case ModelFormat.SAFETENSORS:
                    raise NotImplementedError
                case ModelFormat.INTERNAL:
                    self.__save_internal(model, output_model_destination)
