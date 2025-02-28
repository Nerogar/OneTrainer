from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoModelSaver import HunyuanVideoModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class HunyuanVideoFineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):

    def save(
            self,
            model: HunyuanVideoModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        base_model_saver = HunyuanVideoModelSaver()
        embedding_model_saver = HunyuanVideoEmbeddingSaver()

        base_model_saver.save(model, output_model_format, output_model_destination, dtype)
        embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
