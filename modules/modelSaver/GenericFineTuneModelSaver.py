from modules.model.BaseModel import BaseModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


def make_fine_tune_model_saver(
    model_class: type[BaseModel],
    model_saver_class: type,
    embedding_saver_class: type | None,
):
    class GenericFineTuneModelSaver(
        BaseModelSaver,
        InternalModelSaverMixin,
    ):
        def __init__(self):
            super().__init__()

        def save(
                self,
                model: model_class,
                model_type: ModelType,
                output_model_format: ModelFormat,
                output_model_destination: str,
                dtype: torch.dtype | None,
        ):
            base_model_saver = model_saver_class()
            base_model_saver.save(model, output_model_format, output_model_destination, dtype)

            if embedding_saver_class is not None:
                embedding_model_saver = embedding_saver_class()
                embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

            if output_model_format == ModelFormat.INTERNAL:
                self._save_internal_data(model, output_model_destination)

    return GenericFineTuneModelSaver
