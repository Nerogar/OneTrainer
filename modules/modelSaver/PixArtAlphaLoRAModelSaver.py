import torch

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver
from modules.modelSaver.pixartAlpha.PixArtAlphaLoRASaver import PixArtAlphaLoRASaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class PixArtAlphaLoRAModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):

    def save(
            self,
            model: PixArtAlphaModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        lora_model_saver = PixArtAlphaLoRASaver()
        embedding_model_saver = PixArtAlphaEmbeddingSaver()

        lora_model_saver.save(model, output_model_format, output_model_destination, dtype)
        if not model.train_config.bundle_additional_embeddings:
            embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
