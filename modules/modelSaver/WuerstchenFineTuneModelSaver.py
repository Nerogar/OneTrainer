from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenModelSaver import WuerstchenModelSaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class WuerstchenFineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: WuerstchenModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        base_model_saver = WuerstchenModelSaver()
        embedding_model_saver = WuerstchenEmbeddingSaver()

        base_model_saver.save(model, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
