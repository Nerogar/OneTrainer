from modules.model.BaseModel import BaseModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util import factory
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch


def make_lora_model_saver(
    model_types: list[ModelType] | ModelType,
    model_class: type[BaseModel],
    lora_saver_class: type,
    embedding_saver_class: type | None,
):
    if not isinstance(model_types, list):
        model_types = [model_types]

    class GenericLoRAModelSaver(
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
            lora_model_saver = lora_saver_class()
            lora_model_saver.save(model, output_model_format, output_model_destination, dtype)

            if embedding_saver_class is not None:
                embedding_model_saver = embedding_saver_class()
                if not model.train_config.bundle_additional_embeddings or output_model_format == ModelFormat.INTERNAL:
                    embedding_model_saver.save_multiple(model, output_model_format, output_model_destination, dtype)

            if output_model_format == ModelFormat.INTERNAL:
                self._save_internal_data(model, output_model_destination)

    for model_type in model_types:
        factory.register(BaseModelSaver, GenericLoRAModelSaver, model_type, TrainingMethod.LORA)

    return GenericLoRAModelSaver
