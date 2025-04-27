
from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoLoRASaver import HunyuanVideoLoRASaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class HunyuanVideoLoRAModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: HunyuanVideoModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        lora_model_saver = HunyuanVideoLoRASaver()
        embedding_model_saver = HunyuanVideoEmbeddingSaver()

        lora_model_saver.save(model, output_model_format, output_model_destination, dtype)
        if not model.train_config.bundle_additional_embeddings or output_model_format == ModelFormat.INTERNAL:
            embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
