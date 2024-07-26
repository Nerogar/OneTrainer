import torch

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import StableDiffusion3EmbeddingSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class StableDiffusion3EmbeddingModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):

    def save(
            self,
            model: StableDiffusion3Model,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding_model_saver = StableDiffusion3EmbeddingSaver()

        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_single(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
