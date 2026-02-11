from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util import factory
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


def make_lora_model_loader(
    model_spec_map: dict[ModelType, str],
    model_class: type[BaseModel],
    model_loader_class: type,
    embedding_loader_class: type | None,
    lora_loader_class: type,
):
    class GenericLoRAModelLoader(
        BaseModelLoader,
        ModelSpecModelLoaderMixin,
        InternalModelLoaderMixin,
    ):
        def __init__(self):
            super().__init__()

        def _default_model_spec_name(
                self,
                model_type: ModelType,
        ) -> str | None:
            return model_spec_map.get(model_type)

        def load(
                self,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ) -> model_class | None:
            base_model_loader = model_loader_class()
            lora_model_loader = lora_loader_class()
            if embedding_loader_class is not None:
                embedding_loader = embedding_loader_class()

            model = model_class(model_type=model_type)
            self._load_internal_data(model, model_names.lora)
            model.model_spec = self._load_default_model_spec(model_type)

            if model_names.base_model is not None:
                base_model_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            lora_model_loader.load(model, model_names)
            if embedding_loader_class is not None:
                embedding_loader.load(model, model_names.lora, model_names)

            return model

    for model_type in model_spec_map:
        factory.register(BaseModelLoader, GenericLoRAModelLoader, model_type, TrainingMethod.LORA)
    return GenericLoRAModelLoader
