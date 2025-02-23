from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class StableDiffusionFineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        base_model_saver = StableDiffusionModelSaver()
        embedding_model_saver = StableDiffusionEmbeddingSaver()

        base_model_saver.save(model, model_type, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
