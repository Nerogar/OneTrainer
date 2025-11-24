from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModuleFilter import ModuleFilter


def make_embedding_model_loader(
    model_spec_map: dict[ModelType, str],
    model_class: type[BaseModel],
    model_loader_class: type,
    embedding_loader_class: type,
):
    class GenericEmbeddingModelLoader(
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
                quant_filters: list[ModuleFilter] | None = None,
        ) -> model_class | None:
            base_model_loader = model_loader_class()
            embedding_loader = embedding_loader_class()

            model = model_class(model_type=model_type)
            self._load_internal_data(model, model_names.embedding.model_name)
            model.model_spec = self._load_default_model_spec(model_type)

            if model_names.base_model is not None:
                base_model_loader.load(model, model_type, model_names, weight_dtypes, quant_filters)
            embedding_loader.load(model, model_names.embedding.model_name, model_names)

            return model
    return GenericEmbeddingModelLoader
