from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class HiDreamEmbeddingModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: HiDreamModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        embedding_model_saver = HiDreamEmbeddingSaver()

        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_single(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
