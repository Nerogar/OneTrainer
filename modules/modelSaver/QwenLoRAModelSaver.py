from modules.model.QwenModel import QwenModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.qwen.QwenLoRASaver import QwenLoRASaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class QwenLoRAModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: QwenModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        lora_model_saver = QwenLoRASaver()

        lora_model_saver.save(model, output_model_format, output_model_destination, dtype)
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
