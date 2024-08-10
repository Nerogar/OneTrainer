import torch

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionLoRASaver import StableDiffusionLoRASaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class StableDiffusionLoRAModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):

    def save(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        lora_model_saver = StableDiffusionLoRASaver()
        embedding_model_saver = StableDiffusionEmbeddingSaver()

        lora_model_saver.save(model, output_model_format, output_model_destination, dtype)
        if not model.train_config.bundle_additional_embeddings:
            embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
