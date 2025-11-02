from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.chroma.ChromaEmbeddingLoader import ChromaEmbeddingLoader
from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModuleFilter import ModuleFilter


class ChromaEmbeddingModelLoader(
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
        match model_type:
            case ModelType.CHROMA_1:
                return "resources/sd_model_spec/chroma-embedding.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quant_filters: list[ModuleFilter] | None = None,
    ) -> ChromaModel | None:
        base_model_loader = ChromaModelLoader()
        embedding_loader = ChromaEmbeddingLoader()

        model = ChromaModel(model_type=model_type)
        self._load_internal_data(model, model_names.embedding.model_name)
        model.model_spec = self._load_default_model_spec(model_type)

        if model_names.base_model is not None:
            base_model_loader.load(model, model_type, model_names, weight_dtypes, quant_filters)
        embedding_loader.load(model, model_names.embedding.model_name, model_names)

        return model
