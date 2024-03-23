import json
import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.ClipEmbeddingModelSaverMixin import ClipEmbeddingModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class StableDiffusionXLEmbeddingModelSaver(
    BaseModelSaver,
    ClipEmbeddingModelSaverMixin,
):

    def __save_ckpt(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        text_encoder_1_vector_cpu = self._get_embedding_vector(
            model.tokenizer_1,
            model.text_encoder_1,
            model.embeddings[0].text_tokens,
        ).to("cpu", dtype)

        text_encoder_2_vector_cpu = self._get_embedding_vector(
            model.tokenizer_2,
            model.text_encoder_2,
            model.embeddings[0].text_tokens,
        ).to("cpu", dtype)

        torch.save(
            {
                "clip_l": text_encoder_1_vector_cpu,
                "clip_g": text_encoder_2_vector_cpu,
            },
            destination
        )

    def __save_safetensors(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        text_encoder_1_vector_cpu = self._get_embedding_vector(
            model.tokenizer_1,
            model.text_encoder_1,
            model.embeddings[0].text_tokens,
        ).to("cpu", dtype)

        text_encoder_2_vector_cpu = self._get_embedding_vector(
            model.tokenizer_2,
            model.text_encoder_2,
            model.embeddings[0].text_tokens,
        ).to("cpu", dtype)

        save_file(
            {
                "clip_l": text_encoder_1_vector_cpu,
                "clip_g": text_encoder_2_vector_cpu,
            },
            destination
        )

    def __save_internal(
            self,
            model: StableDiffusionXLModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        # embedding
        self.__save_safetensors(
            model,
            os.path.join(destination, "embedding", "embedding.safetensors"),
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
