from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import StableDiffusionXLEmbeddingSaver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLModelSaver import StableDiffusionXLModelSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class StableDiffusionXLFineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        base_model_saver = StableDiffusionXLModelSaver()
        embedding_model_saver = StableDiffusionXLEmbeddingSaver()

        base_model_saver.save(model, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
